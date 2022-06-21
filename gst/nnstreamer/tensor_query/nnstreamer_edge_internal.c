/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   nnstreamer_edge_internal.c
 * @date   6 April 2022
 * @brief  Common library to support communication among devices.
 * @see    https://github.com/nnstreamer/nnstreamer
 * @author Gichan Jang <gichan2.jang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "nnstreamer_edge_common.h"
#include "nnstreamer_edge_internal.h"

#define N_BACKLOG 10
#define DEFAULT_TIMEOUT_SEC 10
#define _STR_NULL(str) ((str) ? (str) : "(NULL)")

/**
 * @brief enum for nnstreamer edge query commands.
 */
typedef enum
{
  _NNS_EDGE_CMD_ERROR = 0,
  _NNS_EDGE_CMD_TRANSFER_DATA,
  _NNS_EDGE_CMD_HOST_INFO,
  _NNS_EDGE_CMD_CAPABILITY,
  _NNS_EDGE_CMD_END
} nns_edge_cmd_e;

/**
 * @brief Structure for edge command info. It should be fixed size.
 */
typedef struct
{
  nns_edge_cmd_e cmd;
  int64_t client_id;

  /* memory info */
  uint32_t num;
  size_t mem_size[NNS_EDGE_DATA_LIMIT];
} nns_edge_cmd_info_s;

/**
 * @brief Structure for edge command and buffers.
 */
typedef struct
{
  nns_edge_cmd_info_s info;
  void *mem[NNS_EDGE_DATA_LIMIT];
} nns_edge_cmd_s;

/**
 * @brief Data structure for edge TCP connection.
 */
typedef struct
{
  char *ip;
  int port;
  int8_t running;
  pthread_t msg_thread;
  GSocket *socket;
  GCancellable *cancellable;
} nns_edge_conn_s;

/**
 * @brief Data structure for connection data.
 */
typedef struct
{
  nns_edge_conn_s *src_conn;
  nns_edge_conn_s *sink_conn;
  int64_t id;
} nns_edge_conn_data_s;

/**
 * @brief Structures for thread data of message handling.
 */
typedef struct
{
  nns_edge_handle_s *eh;
  int64_t client_id;
  nns_edge_conn_s *conn;
} nns_edge_thread_data_s;

static bool _nns_edge_close_connection (nns_edge_conn_s * conn);
static int _nns_edge_create_message_thread (nns_edge_handle_s * eh,
    nns_edge_conn_s * conn, int64_t client_id);
static int _nns_edge_tcp_connect (nns_edge_handle_s * eh, const char *ip,
    int port);

/**
 * @brief Send data to connected socket.
 */
static bool
_send_raw_data (GSocket * socket, void *data, size_t size,
    GCancellable * cancellable)
{
  size_t bytes_sent = 0;
  ssize_t rret;
  GError *err = NULL;

  while (bytes_sent < size) {
    rret = g_socket_send (socket, (char *) data + bytes_sent,
        size - bytes_sent, cancellable, &err);

    if (rret == 0) {
      nns_edge_loge ("Connection closed.");
      return false;
    }

    if (rret < 0) {
      nns_edge_loge ("Error while sending data (%s).", err->message);
      g_clear_error (&err);
      return false;
    }

    bytes_sent += rret;
  }

  return true;
}

/**
 * @brief Receive data from connected socket.
 */
static bool
_receive_raw_data (GSocket * socket, void *data, size_t size,
    GCancellable * cancellable)
{
  size_t bytes_received = 0;
  ssize_t rret;
  GError *err = NULL;

  while (bytes_received < size) {
    rret = g_socket_receive (socket, (char *) data + bytes_received,
        size - bytes_received, cancellable, &err);

    if (rret == 0) {
      nns_edge_loge ("Connection closed.");
      return false;
    }

    if (rret < 0) {
      nns_edge_loge ("Failed to read from socket (%s).", err->message);
      g_clear_error (&err);
      return false;
    }

    bytes_received += rret;
  }

  return true;
}

/**
 * @brief Parse string and get host IP:port.
 */
static void
_parse_host_str (const char *host, char **ip, int *port)
{
  char *p = g_strrstr (host, ":");

  if (p) {
    *ip = g_strndup (host, (p - host));
    *port = (int) g_ascii_strtoll (p + 1, NULL, 10);
  }
}

/**
 * @brief Get host string (IP:port).
 */
static void
_get_host_str (const char *ip, const int port, char **host)
{
  *host = g_strdup_printf ("%s:%d", ip, port);
}

/**
 * @brief initialize edge command.
 */
static void
_nns_edge_cmd_init (nns_edge_cmd_s * cmd, nns_edge_cmd_e c, int64_t cid)
{
  if (!cmd)
    return;

  memset (cmd, 0, sizeof (nns_edge_cmd_s));
  cmd->info.cmd = c;
  cmd->info.client_id = cid;
}

/**
 * @brief Clear allocated memory in edge command.
 */
static void
_nns_edge_cmd_clear (nns_edge_cmd_s * cmd)
{
  unsigned int i;

  if (!cmd)
    return;

  for (i = 0; i < cmd->info.num; i++) {
    if (cmd->mem[i])
      free (cmd->mem[i]);
    cmd->mem[i] = NULL;
  }
}

/**
 * @brief Send edge command to connected device.
 */
static int
_nns_edge_cmd_send (nns_edge_conn_s * conn, nns_edge_cmd_s * cmd)
{
  unsigned int n;

  if (!conn || !cmd)
    return NNS_EDGE_ERROR_INVALID_PARAMETER;

  if (!_send_raw_data (conn->socket, &cmd->info,
          sizeof (nns_edge_cmd_info_s), conn->cancellable)) {
    nns_edge_loge ("Failed to send command to socket.");
    return NNS_EDGE_ERROR_IO;
  }

  for (n = 0; n < cmd->info.num; n++) {
    if (!_send_raw_data (conn->socket, cmd->mem[n],
            cmd->info.mem_size[n], conn->cancellable)) {
      nns_edge_loge ("Failed to send %uth memory to socket.", n);
      return NNS_EDGE_ERROR_IO;
    }
  }

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Receive edge command from connected device.
 */
static int
_nns_edge_cmd_receive (nns_edge_conn_s * conn, nns_edge_cmd_s * cmd)
{
  unsigned int i, n;
  int ret = NNS_EDGE_ERROR_NONE;

  if (!conn || !cmd)
    return NNS_EDGE_ERROR_INVALID_PARAMETER;

  if (!_receive_raw_data (conn->socket, &cmd->info,
          sizeof (nns_edge_cmd_info_s), conn->cancellable)) {
    nns_edge_loge ("Failed to receive command from socket.");
    return NNS_EDGE_ERROR_IO;
  }

  nns_edge_logd ("Received command:%d (num:%u)", cmd->info.cmd, cmd->info.num);

  for (n = 0; n < cmd->info.num; n++) {
    cmd->mem[n] = malloc (cmd->info.mem_size[n]);
    if (!cmd->mem[n]) {
      nns_edge_loge ("Failed to allocate memory to receive data from socket.");
      ret = NNS_EDGE_ERROR_OUT_OF_MEMORY;
      break;
    }

    if (!_receive_raw_data (conn->socket, cmd->mem[n],
            cmd->info.mem_size[n], conn->cancellable)) {
      nns_edge_loge ("Failed to receive %uth memory from socket.", n++);
      ret = NNS_EDGE_ERROR_IO;
      break;
    }
  }

  if (ret != NNS_EDGE_ERROR_NONE) {
    for (i = 0; i < n; i++) {
      free (cmd->mem[i]);
      cmd->mem[i] = NULL;
    }
  }

  return ret;
}

/**
 * @brief Internal function to invoke event callback.
 * @note This function should be called with handle lock.
 */
static int
_nns_edge_invoke_event_cb (nns_edge_handle_s * eh, nns_edge_event_e event,
    void *data, size_t data_len, nns_edge_data_destroy_cb destroy_cb)
{
  nns_edge_event_h event_h;
  int ret;

  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  /* If event callback is null, return ok. */
  if (!eh->event_cb) {
    nns_edge_logw ("The event callback is null, do nothing!");
    return NNS_EDGE_ERROR_NONE;
  }

  ret = nns_edge_event_create (event, &event_h);
  if (ret != NNS_EDGE_ERROR_NONE) {
    nns_edge_loge ("Failed to create new edge event.");
    return ret;
  }

  if (data) {
    ret = nns_edge_event_set_data (event_h, data, data_len, destroy_cb);
    if (ret != NNS_EDGE_ERROR_NONE) {
      nns_edge_loge ("Failed to handle edge event due to invalid event data.");
      goto error;
    }
  }

  ret = eh->event_cb (event_h, eh->user_data);
  if (ret != NNS_EDGE_ERROR_NONE) {
    nns_edge_loge ("The event callback returns error.");
  }

error:
  nns_edge_event_destroy (event_h);
  return ret;
}

/**
 * @brief Get nnstreamer-edge connection data.
 * @note This function should be called with handle lock.
 */
static nns_edge_conn_data_s *
_nns_edge_get_conn (nns_edge_handle_s * eh, int64_t client_id)
{
  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return NULL;
  }

  return g_hash_table_lookup (eh->conn_table, GUINT_TO_POINTER (client_id));
}

/**
 * @brief Get nnstreamer-edge connection data.
 * @note This function should be called with handle lock.
 */
static nns_edge_conn_data_s *
_nns_edge_add_conn (nns_edge_handle_s * eh, int64_t client_id)
{
  nns_edge_conn_data_s *data = NULL;

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return NULL;
  }

  data = g_hash_table_lookup (eh->conn_table, GUINT_TO_POINTER (client_id));

  if (NULL == data) {
    data = (nns_edge_conn_data_s *) malloc (sizeof (nns_edge_conn_data_s));
    if (NULL == data) {
      nns_edge_loge ("Failed to allocate memory for connection data.");
      return NULL;
    }

    memset (data, 0, sizeof (nns_edge_conn_data_s));
    data->id = client_id;

    g_hash_table_insert (eh->conn_table, GUINT_TO_POINTER (client_id), data);
  }

  return data;
}

/**
 * @brief Remove nnstreamer-edge connection data. This will be called when removing connection data from hash table.
 */
static void
_nns_edge_remove_conn (gpointer data)
{
  nns_edge_conn_data_s *cdata = (nns_edge_conn_data_s *) data;

  if (cdata) {
    _nns_edge_close_connection (cdata->src_conn);
    _nns_edge_close_connection (cdata->sink_conn);

    g_free (cdata);
  }
}

/**
 * @brief Internal function to check connection.
 */
static bool
_nns_edge_check_connection (nns_edge_conn_s * conn)
{
  size_t size;
  GIOCondition condition;

  if (!conn)
    return false;

  condition = g_socket_condition_check (conn->socket,
      G_IO_IN | G_IO_PRI | G_IO_ERR | G_IO_HUP);
  size = g_socket_get_available_bytes (conn->socket);

  if (condition && size <= 0) {
    nns_edge_logw ("Socket is not available, possibly EOS.");
    return false;
  }

  return true;
}

/**
 * @brief Get socket address
 */
static bool
_nns_edge_get_saddr (const char *ip, const int port,
    GCancellable * cancellable, GSocketAddress ** saddr)
{
  GError *err = NULL;
  GInetAddress *addr;

  /* look up name if we need to */
  addr = g_inet_address_new_from_string (ip);
  if (!addr) {
    GList *results;
    GResolver *resolver;
    resolver = g_resolver_get_default ();
    results = g_resolver_lookup_by_name (resolver, ip, cancellable, &err);
    if (!results) {
      if (g_error_matches (err, G_IO_ERROR, G_IO_ERROR_CANCELLED)) {
        nns_edge_loge ("Failed to resolve ip, name resolver is cancelled.");
      } else {
        nns_edge_loge ("Failed to resolve ip '%s': %s", ip, err->message);
      }
      g_clear_error (&err);
      g_object_unref (resolver);
      return false;
    }
    /** @todo Try with the second address if the first fails */
    addr = G_INET_ADDRESS (g_object_ref (results->data));
    g_resolver_free_addresses (results);
    g_object_unref (resolver);
  }

  *saddr = g_inet_socket_address_new (addr, port);
  g_object_unref (addr);

  return true;
}

/**
 * @brief Get registered handle. If not registered, create new handle and register it.
 */
int
nns_edge_create_handle (const char *id, const char *topic, nns_edge_h * edge_h)
{
  nns_edge_handle_s *eh;

  if (!id || *id == '\0') {
    nns_edge_loge ("Invalid param, given ID is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!topic || *topic == '\0') {
    nns_edge_loge ("Invalid param, given topic is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!edge_h) {
    nns_edge_loge ("Invalid param, edge_h should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  /**
   * @todo manage edge handles
   * 1. consider adding hash table or list to manage edge handles.
   * 2. compare topic and return error if existing topic in handle is different.
   */
  eh = (nns_edge_handle_s *) malloc (sizeof (nns_edge_handle_s));
  if (!eh) {
    nns_edge_loge ("Failed to allocate memory for edge handle.");
    return NNS_EDGE_ERROR_OUT_OF_MEMORY;
  }

  memset (eh, 0, sizeof (nns_edge_handle_s));
  nns_edge_lock_init (eh);
  eh->magic = NNS_EDGE_MAGIC;
  eh->id = g_strdup (id);
  eh->topic = g_strdup (topic);
  eh->protocol = NNS_EDGE_PROTOCOL_TCP;
  eh->is_server = true;
  eh->recv_ip = g_strdup ("localhost");
  eh->recv_port = 0;
  eh->caps_str = NULL;

  /* Connection data for each client ID. */
  eh->conn_table = g_hash_table_new_full (g_direct_hash, g_direct_equal, NULL,
      _nns_edge_remove_conn);

  *edge_h = eh;
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief [TCP] Callback for src socket listener that pushes socket to the queue
 */
static void
accept_socket_async_cb (GObject * source, GAsyncResult * result,
    gpointer user_data)
{
  GSocketListener *socket_listener = G_SOCKET_LISTENER (source);
  GSocket *socket = NULL;
  GError *err = NULL;
  nns_edge_handle_s *eh = (nns_edge_handle_s *) user_data;
  nns_edge_conn_s *conn = NULL;
  nns_edge_cmd_s cmd;
  bool done = false;
  char *connected_ip = NULL;
  int connected_port = 0;
  nns_edge_conn_data_s *conn_data = NULL;
  int64_t client_id;
  int ret;

  socket =
      g_socket_listener_accept_socket_finish (socket_listener, result, NULL,
      &err);

  if (!socket) {
    nns_edge_loge ("Failed to get socket: %s", err->message);
    g_clear_error (&err);
    goto error;
  }
  g_socket_set_timeout (socket, DEFAULT_TIMEOUT_SEC);

  /* create socket with connection */
  conn = (nns_edge_conn_s *) malloc (sizeof (nns_edge_conn_s));
  if (!conn) {
    nns_edge_loge ("Failed to allocate edge connection");
    goto error;
  }

  memset (conn, 0, sizeof (nns_edge_conn_s));
  conn->socket = socket;
  conn->cancellable = g_cancellable_new ();

  /* setting TCP_NODELAY to true in order to avoid packet batching as known as Nagle's algorithm */
  if (!g_socket_set_option (socket, IPPROTO_TCP, TCP_NODELAY, true, &err)) {
    nns_edge_loge ("Failed to set socket TCP_NODELAY option: %s", err->message);
    g_clear_error (&err);
    goto error;
  }

  /* Send capability and info to check compatibility. */
  client_id = eh->is_server ? g_get_monotonic_time () : eh->client_id;

  _nns_edge_cmd_init (&cmd, _NNS_EDGE_CMD_CAPABILITY, client_id);
  cmd.info.num = 1;
  cmd.info.mem_size[0] = strlen (eh->caps_str) + 1;
  cmd.mem[0] = eh->caps_str;

  ret = _nns_edge_cmd_send (conn, &cmd);
  if (ret != NNS_EDGE_ERROR_NONE) {
    nns_edge_loge ("Failed to send capability.");
    goto error;
  }

  /* Receive ip and port from destination. */
  ret = _nns_edge_cmd_receive (conn, &cmd);
  if (ret != NNS_EDGE_ERROR_NONE) {
    nns_edge_loge ("Failed to receive node info.");
    goto error;
  }

  if (cmd.info.cmd != _NNS_EDGE_CMD_HOST_INFO) {
    nns_edge_loge ("Failed to get host info.");
    _nns_edge_cmd_clear (&cmd);
    goto error;
  }

  _parse_host_str (cmd.mem[0], &connected_ip, &connected_port);
  _nns_edge_cmd_clear (&cmd);

  ret = _nns_edge_create_message_thread (eh, conn, client_id);
  if (ret != NNS_EDGE_ERROR_NONE) {
    nns_edge_loge ("Failed to create message handle thread.");
    goto error;
  }

  conn_data = _nns_edge_add_conn (eh, client_id);
  if (conn_data) {
    /* Close old connection and set new one. */
    _nns_edge_close_connection (conn_data->src_conn);
    conn_data->src_conn = conn;
    done = true;
  }

error:
  if (done) {
    if (eh->is_server) {
      _nns_edge_tcp_connect (eh, connected_ip, connected_port);
    }
  } else {
    _nns_edge_close_connection (conn);
  }

  g_socket_listener_accept_socket_async (socket_listener,
      eh->cancellable, (GAsyncReadyCallback) accept_socket_async_cb, eh);

  g_free (connected_ip);
}

/**
 * @brief Get available port number.
 */
static int
_get_available_port (void)
{
  struct sockaddr_in sin;
  int port = 0, sock;
  socklen_t len = sizeof (struct sockaddr);

  sin.sin_family = AF_INET;
  sin.sin_addr.s_addr = INADDR_ANY;
  sock = socket (AF_INET, SOCK_STREAM, 0);
  sin.sin_port = port;
  if (bind (sock, (struct sockaddr *) &sin, sizeof (struct sockaddr)) == 0) {
    getsockname (sock, (struct sockaddr *) &sin, &len);
    port = ntohs (sin.sin_port);
    nns_edge_logi ("Available port number: %d", port);
  }
  close (sock);

  return port;
}

/**
 * @brief Initialize the nnstreamer edge handle.
 */
int
nns_edge_start (nns_edge_h edge_h, bool is_server)
{
  GSocketAddress *saddr = NULL;
  GError *err = NULL;
  int ret = 0;
  nns_edge_handle_s *eh;

  eh = (nns_edge_handle_s *) edge_h;
  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  eh->is_server = is_server;
  if (!is_server && 0 == eh->recv_port)
    eh->recv_port = _get_available_port ();

  /** Initialize server src data. */
  eh->cancellable = g_cancellable_new ();
  eh->listener = g_socket_listener_new ();
  g_socket_listener_set_backlog (eh->listener, N_BACKLOG);

  if (!_nns_edge_get_saddr (eh->recv_ip, eh->recv_port, eh->cancellable,
          &saddr)) {
    nns_edge_loge ("Failed to get socket address");
    ret = NNS_EDGE_ERROR_CONNECTION_FAILURE;
    goto error;
  }
  if (!g_socket_listener_add_address (eh->listener, saddr,
          G_SOCKET_TYPE_STREAM, G_SOCKET_PROTOCOL_TCP, NULL, NULL, &err)) {
    nns_edge_loge ("Failed to add address: %s", err->message);
    g_clear_error (&err);
    ret = NNS_EDGE_ERROR_CONNECTION_FAILURE;
    goto error;
  }
  g_object_unref (saddr);
  saddr = NULL;

  g_socket_listener_accept_socket_async (eh->listener,
      eh->cancellable, (GAsyncReadyCallback) accept_socket_async_cb, eh);

error:
  if (saddr)
    g_object_unref (saddr);

  nns_edge_unlock (eh);
  return ret;
}

/**
 * @brief Release the given handle.
 */
int
nns_edge_release_handle (nns_edge_h edge_h)
{
  nns_edge_handle_s *eh;

  eh = (nns_edge_handle_s *) edge_h;
  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  eh->magic = NNS_EDGE_MAGIC_DEAD;
  eh->event_cb = NULL;
  eh->user_data = NULL;
  g_free (eh->id);
  g_free (eh->topic);
  g_free (eh->ip);
  g_free (eh->recv_ip);
  g_hash_table_destroy (eh->conn_table);

  nns_edge_unlock (eh);
  nns_edge_lock_destroy (eh);
  g_free (eh);

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Set the event callback.
 */
int
nns_edge_set_event_callback (nns_edge_h edge_h, nns_edge_event_cb cb,
    void *user_data)
{
  nns_edge_handle_s *eh;
  int ret;

  eh = (nns_edge_handle_s *) edge_h;
  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  ret = _nns_edge_invoke_event_cb (eh, NNS_EDGE_EVENT_CALLBACK_RELEASED,
      NULL, 0, NULL);
  if (ret != NNS_EDGE_ERROR_NONE) {
    nns_edge_loge ("Failed to set new event callback.");
    nns_edge_unlock (eh);
    return ret;
  }

  eh->event_cb = cb;
  eh->user_data = user_data;

  nns_edge_unlock (eh);
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Connect to requested socket using TCP.
 */
static bool
_nns_edge_connect_socket (nns_edge_conn_s * conn)
{
  GError *err = NULL;
  GSocketAddress *saddr = NULL;
  bool ret = false;

  if (!_nns_edge_get_saddr (conn->ip, conn->port, conn->cancellable, &saddr)) {
    nns_edge_loge ("Failed to get socket address");
    return ret;
  }

  /* create sending client socket */
  conn->socket =
      g_socket_new (g_socket_address_get_family (saddr), G_SOCKET_TYPE_STREAM,
      G_SOCKET_PROTOCOL_TCP, &err);

  if (!conn->socket) {
    nns_edge_loge ("Failed to create new socket");
    goto done;
  }

  /* setting TCP_NODELAY to true in order to avoid packet batching as known as Nagle's algorithm */
  if (!g_socket_set_option (conn->socket, IPPROTO_TCP, TCP_NODELAY, true, &err)) {
    nns_edge_loge ("Failed to set socket TCP_NODELAY option: %s", err->message);
    goto done;
  }

  if (!g_socket_connect (conn->socket, saddr, conn->cancellable, &err)) {
    if (g_error_matches (err, G_IO_ERROR, G_IO_ERROR_CANCELLED)) {
      nns_edge_logd ("Cancelled connecting");
    } else {
      nns_edge_loge ("Failed to connect to host, %s:%d", conn->ip, conn->port);
    }
    goto done;
  }

  /* now connected to the requested socket */
  ret = true;

done:
  g_object_unref (saddr);
  g_clear_error (&err);
  return ret;
}

/**
 * @brief [TCP] Receive buffer from the client
 * @param[in] conn connection info
 */
static void *
_message_handler (void *thread_data)
{
  nns_edge_thread_data_s *_tdata = (nns_edge_thread_data_s *) thread_data;
  nns_edge_handle_s *eh;
  nns_edge_conn_s *conn;
  nns_edge_cmd_s cmd;
  int64_t client_id;
  char *val;
  int ret;

  if (!_tdata) {
    nns_edge_loge ("Internal error, thread data is null.");
    return NULL;
  }

  eh = (nns_edge_handle_s *) _tdata->eh;
  conn = _tdata->conn;
  client_id = _tdata->client_id;
  g_free (_tdata);

  while (conn->running) {
    nns_edge_data_h data_h;
    unsigned int i;

    /* Validate edge handle */
    if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
      nns_edge_loge ("The edge handle is invalid, it would be expired.");
      break;
    }

    if (!_nns_edge_check_connection (conn))
      break;

    /** Receive data from the client */
    ret = _nns_edge_cmd_receive (conn, &cmd);
    if (ret != NNS_EDGE_ERROR_NONE) {
      nns_edge_loge ("Failed to receive data from the connected node.");
      break;
    }

    if (cmd.info.cmd == _NNS_EDGE_CMD_ERROR) {
      nns_edge_loge ("Received error, stop msg thread.");
      _nns_edge_cmd_clear (&cmd);
      break;
    } else if (cmd.info.cmd != _NNS_EDGE_CMD_TRANSFER_DATA) {
      /** @todo handle other cmd later */
      _nns_edge_cmd_clear (&cmd);
      continue;
    }

    ret = nns_edge_data_create (&data_h);
    if (ret != NNS_EDGE_ERROR_NONE) {
      nns_edge_loge ("Failed to create data handle in msg thread.");
      _nns_edge_cmd_clear (&cmd);
      continue;
    }

    /* Set client ID in edge data */
    val = g_strdup_printf ("%ld", (long int) client_id);
    nns_edge_data_set_info (data_h, "client_id", val);
    g_free (val);

    for (i = 0; i < cmd.info.num; i++) {
      nns_edge_data_add (data_h, cmd.mem[i], cmd.info.mem_size[i], NULL);
    }

    ret = _nns_edge_invoke_event_cb (eh, NNS_EDGE_EVENT_NEW_DATA_RECEIVED,
        data_h, sizeof (data_h), NULL);
    if (ret != NNS_EDGE_ERROR_NONE) {
      /* Try to get next request if server does not accept data from client. */
      nns_edge_logw ("The server does not accept data from client.");
    }

    nns_edge_data_destroy (data_h);
    _nns_edge_cmd_clear (&cmd);
  }

  conn->running = 0;
  return NULL;
}

/**
 * @brief Create message handle thread.
 */
static int
_nns_edge_create_message_thread (nns_edge_handle_s * eh, nns_edge_conn_s * conn,
    int64_t client_id)
{
  pthread_attr_t attr;
  int tid;
  nns_edge_thread_data_s *thread_data = NULL;

  thread_data =
      (nns_edge_thread_data_s *) malloc (sizeof (nns_edge_thread_data_s));
  if (!thread_data) {
    nns_edge_loge ("Failed to allocate edge thread data.");
    return NNS_EDGE_ERROR_OUT_OF_MEMORY;
  }

   /** Create message receving thread */
  pthread_attr_init (&attr);
  pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_DETACHED);
  conn->running = 1;
  thread_data->eh = eh;
  thread_data->conn = conn;
  thread_data->client_id = client_id;

  tid =
      pthread_create (&conn->msg_thread, &attr, _message_handler, thread_data);
  pthread_attr_destroy (&attr);

  if (tid < 0) {
    nns_edge_loge ("Failed to create message handler thread.");
    conn->running = 0;
    g_free (thread_data);
    return NNS_EDGE_ERROR_IO;
  }

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Connect to the destination node usig TCP.
 */
static int
_nns_edge_tcp_connect (nns_edge_handle_s * eh, const char *ip, int port)
{
  nns_edge_conn_s *conn = NULL;
  nns_edge_conn_data_s *conn_data;
  nns_edge_cmd_s cmd;
  char *host;
  int64_t client_id;
  bool done = false;
  int ret;

  conn = (nns_edge_conn_s *) malloc (sizeof (nns_edge_conn_s));
  if (!conn) {
    nns_edge_loge ("Failed to allocate client data.");
    goto error;
  }

  memset (conn, 0, sizeof (nns_edge_conn_s));
  conn->ip = g_strdup (ip);
  conn->port = port;
  conn->cancellable = g_cancellable_new ();

  if (!_nns_edge_connect_socket (conn)) {
    goto error;
  }

  /* Get destination capability. */
  ret = _nns_edge_cmd_receive (conn, &cmd);
  if (ret != NNS_EDGE_ERROR_NONE) {
    nns_edge_loge ("Failed to receive capability.");
    goto error;
  }

  if (cmd.info.cmd != _NNS_EDGE_CMD_CAPABILITY) {
    nns_edge_loge ("Failed to get capability.");
    _nns_edge_cmd_clear (&cmd);
    goto error;
  }

  client_id = eh->client_id = cmd.info.client_id;

  /* Check compatibility. */
  ret = _nns_edge_invoke_event_cb (eh, NNS_EDGE_EVENT_CAPABILITY,
      cmd.mem[0], cmd.info.mem_size[0], NULL);
  _nns_edge_cmd_clear (&cmd);

  if (ret != NNS_EDGE_ERROR_NONE) {
    nns_edge_loge ("The event returns error, capability is not acceptable.");
    _nns_edge_cmd_init (&cmd, _NNS_EDGE_CMD_ERROR, client_id);
  } else {
    /* Send ip and port to destination. */
    _nns_edge_cmd_init (&cmd, _NNS_EDGE_CMD_HOST_INFO, client_id);

    _get_host_str (eh->recv_ip, eh->recv_port, &host);
    cmd.info.num = 1;
    cmd.info.mem_size[0] = strlen (host) + 1;
    cmd.mem[0] = host;
  }

  ret = _nns_edge_cmd_send (conn, &cmd);
  _nns_edge_cmd_clear (&cmd);

  if (ret != NNS_EDGE_ERROR_NONE) {
    nns_edge_loge ("Failed to send host info.");
    goto error;
  }

  conn_data = _nns_edge_add_conn (eh, client_id);
  if (conn_data) {
    /* Close old connection and set new one. */
    _nns_edge_close_connection (conn_data->sink_conn);
    conn_data->sink_conn = conn;
    done = true;
  }

error:
  if (!done) {
    _nns_edge_close_connection (conn);
    return NNS_EDGE_ERROR_CONNECTION_FAILURE;
  }

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Connect to the destination node.
 */
int
nns_edge_connect (nns_edge_h edge_h, nns_edge_protocol_e protocol,
    const char *ip, int port)
{
  nns_edge_handle_s *eh;
  int ret;

  eh = (nns_edge_handle_s *) edge_h;
  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!ip || *ip == '\0') {
    nns_edge_loge ("Invalid param, given IP is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!eh->event_cb) {
    nns_edge_loge ("NNStreamer-edge event callback is not registered.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_CONNECTION_FAILURE;
  }

  eh->is_server = false;
  eh->protocol = protocol;

  /** Connect to info channel. */
  ret = _nns_edge_tcp_connect (eh, ip, port);
  if (ret != NNS_EDGE_ERROR_NONE) {
    nns_edge_loge ("Failed to connect to %s:%d", ip, port);
  }

  nns_edge_unlock (eh);
  return ret;
}

/**
 * @brief Close connection
 */
static bool
_nns_edge_close_connection (nns_edge_conn_s * conn)
{
  GError *err = NULL;

  if (!conn)
    return false;

  if (conn->running) {
    conn->running = 0;
    pthread_join (conn->msg_thread, NULL);
  }

  if (conn->socket) {
    if (!g_socket_close (conn->socket, &err)) {
      nns_edge_loge ("Failed to close socket: %s", err->message);
      g_clear_error (&err);
      return false;
    }
    g_object_unref (conn->socket);
    conn->socket = NULL;
  }

  if (conn->cancellable) {
    g_object_unref (conn->cancellable);
    conn->cancellable = NULL;
  }

  g_free (conn->ip);
  g_free (conn);
  return true;
}

/**
 * @brief Disconnect from the destination node.
 */
int
nns_edge_disconnect (nns_edge_h edge_h)
{
  nns_edge_handle_s *eh;

  eh = (nns_edge_handle_s *) edge_h;
  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  g_hash_table_remove_all (eh->conn_table);

  nns_edge_unlock (eh);
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Publish a message to a given topic.
 */
int
nns_edge_publish (nns_edge_h edge_h, nns_edge_data_h data_h)
{
  nns_edge_handle_s *eh;

  eh = (nns_edge_handle_s *) edge_h;
  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (nns_edge_data_is_valid (data_h) != NNS_EDGE_ERROR_NONE) {
    nns_edge_loge ("Invalid param, given edge data is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  /** @todo update code (publish data) */

  nns_edge_unlock (eh);
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Request result to the server.
 */
int
nns_edge_request (nns_edge_h edge_h, nns_edge_data_h data_h, void *user_data)
{
  nns_edge_handle_s *eh;
  nns_edge_conn_data_s *conn_data;
  nns_edge_cmd_s cmd;
  int ret;
  unsigned int i;

  UNUSED (user_data);
  eh = (nns_edge_handle_s *) edge_h;
  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (nns_edge_data_is_valid (data_h) != NNS_EDGE_ERROR_NONE) {
    nns_edge_loge ("Invalid param, given edge data is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  conn_data = _nns_edge_get_conn (eh, eh->client_id);
  if (!_nns_edge_check_connection (conn_data->sink_conn)) {
    nns_edge_loge ("Failed to request, connection failure.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_CONNECTION_FAILURE;
  }

  _nns_edge_cmd_init (&cmd, _NNS_EDGE_CMD_TRANSFER_DATA, eh->client_id);

  nns_edge_data_get_count (data_h, &cmd.info.num);
  for (i = 0; i < cmd.info.num; i++) {
    nns_edge_data_get (data_h, i, &cmd.mem[i], &cmd.info.mem_size[i]);
  }

  ret = _nns_edge_cmd_send (conn_data->sink_conn, &cmd);

  nns_edge_unlock (eh);
  return ret;
}

/**
 * @brief Subscribe a message to a given topic.
 */
int
nns_edge_subscribe (nns_edge_h edge_h, nns_edge_data_h data_h, void *user_data)
{
  nns_edge_handle_s *eh;

  UNUSED (user_data);
  eh = (nns_edge_handle_s *) edge_h;
  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (nns_edge_data_is_valid (data_h) != NNS_EDGE_ERROR_NONE) {
    nns_edge_loge ("Invalid param, given edge data is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  /** @todo update code (subscribe) */

  nns_edge_unlock (eh);
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Unsubscribe a message to a given topic.
 */
int
nns_edge_unsubscribe (nns_edge_h edge_h)
{
  nns_edge_handle_s *eh;

  eh = (nns_edge_handle_s *) edge_h;
  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  /** @todo update code (unsubscribe) */

  nns_edge_unlock (eh);
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Get the topic of edge handle. Caller should release returned string using free().
 * @todo is this necessary?
 */
int
nns_edge_get_topic (nns_edge_h edge_h, char **topic)
{
  nns_edge_handle_s *eh;

  eh = (nns_edge_handle_s *) edge_h;
  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!topic) {
    nns_edge_loge ("Invalid param, topic should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  *topic = g_strdup (eh->topic);

  nns_edge_unlock (eh);
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Set nnstreamer edge info.
 */
int
nns_edge_set_info (nns_edge_h edge_h, const char *key, const char *value)
{
  nns_edge_handle_s *eh;
  char *ret_str = NULL;

  eh = (nns_edge_handle_s *) edge_h;
  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  /**
   * @todo User handles (replace or append) the capability of edge handle.
   * @todo Change key-value set as json or hash table.
   */
  if (0 == g_ascii_strcasecmp (key, "CAPS")) {
    ret_str = g_strdup_printf ("%s%s", _STR_NULL (eh->caps_str), value);
    g_free (eh->caps_str);
    eh->caps_str = ret_str;
  } else if (0 == g_ascii_strcasecmp (key, "IP")) {
    g_free (eh->recv_ip);
    eh->recv_ip = g_strdup (value);
  } else if (0 == g_ascii_strcasecmp (key, "PORT")) {
    eh->recv_port = g_ascii_strtoll (value, NULL, 10);
  } else if (0 == g_ascii_strcasecmp (key, "TOPIC")) {
    g_free (eh->topic);
    eh->topic = g_strdup (value);
  } else {
    nns_edge_logw ("Failed to set edge info. Unknown key: %s", key);
  }

  nns_edge_unlock (eh);
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Respond to a request.
 */
int
nns_edge_respond (nns_edge_h edge_h, nns_edge_data_h data_h)
{
  nns_edge_handle_s *eh;
  nns_edge_conn_data_s *conn_data;
  nns_edge_cmd_s cmd;
  int64_t client_id;
  char *val;
  int ret;
  unsigned int i;

  eh = (nns_edge_handle_s *) edge_h;
  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (nns_edge_data_is_valid (data_h) != NNS_EDGE_ERROR_NONE) {
    nns_edge_loge ("Invalid param, given edge data is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  ret = nns_edge_data_get_info (data_h, "client_id", &val);
  if (ret != NNS_EDGE_ERROR_NONE) {
    nns_edge_loge ("Cannot find client ID in edge data.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  client_id = g_ascii_strtoll (val, NULL, 10);
  g_free (val);

  conn_data = _nns_edge_get_conn (eh, client_id);
  if (!conn_data) {
    nns_edge_loge ("Cannot find connection, invalid client ID.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  _nns_edge_cmd_init (&cmd, _NNS_EDGE_CMD_TRANSFER_DATA, client_id);

  nns_edge_data_get_count (data_h, &cmd.info.num);
  for (i = 0; i < cmd.info.num; i++) {
    nns_edge_data_get (data_h, i, &cmd.mem[i], &cmd.info.mem_size[i]);
  }

  ret = _nns_edge_cmd_send (conn_data->sink_conn, &cmd);

  nns_edge_unlock (eh);
  return ret;
}