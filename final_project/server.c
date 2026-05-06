/*
 * =============================================================================
 * TCP Multi-Client Chat Server
 * =============================================================================
 * Course:  CS 470 - Final Assignment: Network Socket Programming Project
 * Author:  Noah Alguire
 * Date:    Spring 2026
 *
 * Reference: Beej's Guide to Network Programming
 *            https://beej.us/guide/bgnet/
 *
 * =============================================================================
 * PROTOCOL CHOICE: TCP vs UDP
 * =============================================================================
 * This application uses TCP (Transmission Control Protocol) for the following
 * reasons:
 *
 * 1. RELIABILITY: TCP guarantees that all data sent will be received by the
 *    other end, in order, and without duplication. In a chat application,
 *    losing messages is unacceptable — users expect every message they send
 *    to be delivered.
 *
 * 2. ORDERED DELIVERY: TCP ensures packets arrive in the order they were sent.
 *    Chat conversations must maintain chronological order; out-of-order
 *    messages would make conversations incoherent.
 *
 * 3. CONNECTION-ORIENTED: TCP's connection model lets the server track which
 *    clients are connected and manage their state (usernames, join/leave
 *    notifications). UDP's connectionless nature would require implementing
 *    this tracking manually.
 *
 * 4. FLOW CONTROL: TCP automatically handles flow control, preventing a fast
 *    sender from overwhelming a slow receiver. This matters when one client
 *    sends many messages quickly.
 *
 * UDP would be preferred for applications where speed matters more than
 * reliability (e.g., real-time voice/video, online games with positional
 * updates), but for text-based chat, the slight latency overhead of TCP's
 * reliability mechanisms is negligible and the guarantees are essential.
 *
 * =============================================================================
 * DESCRIPTION
 * =============================================================================
 * This server listens on a configurable port and accepts multiple simultaneous
 * client connections. It uses poll() for I/O multiplexing to handle all clients
 * in a single thread. Features include:
 *
 *   - Nickname support: clients set a username on connect
 *   - Message broadcasting: messages are relayed to all other connected clients
 *   - Join/leave notifications: all clients are notified when users connect
 *     or disconnect
 *   - Private messaging: /msg <user> <text> sends a direct message
 *   - Active user listing: /who command shows connected users
 *   - Graceful shutdown: handles SIGINT for clean server termination
 *
 * Usage: ./server [port]
 *        Default port: 9034
 *
 * Based on the pollserver.c example from Beej's Guide (Section 7.2).
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <poll.h>
#include <time.h>

#define DEFAULT_PORT  "9034"
#define MAX_CLIENTS   64
#define BUFFER_SIZE   4096
#define NICK_SIZE     32
#define MSG_SIZE      (BUFFER_SIZE + NICK_SIZE + 64)

/* Client tracking structure */
typedef struct {
    int  fd;                  /* socket file descriptor, -1 if unused */
    char nickname[NICK_SIZE]; /* display name */
    int  has_nick;            /* 1 if nickname has been set */
} client_t;

static client_t clients[MAX_CLIENTS];
static struct pollfd pfds[MAX_CLIENTS + 1]; /* +1 for the listener */
static int pfd_count = 0;
static volatile sig_atomic_t running = 1;

/* -------------------------------------------------------------------------- */
/* Signal handler for graceful shutdown                                       */
/* -------------------------------------------------------------------------- */
static void handle_sigint(int sig)
{
    (void)sig;
    running = 0;
}

/* -------------------------------------------------------------------------- */
/* Get a human-readable timestamp string                                      */
/* -------------------------------------------------------------------------- */
static void get_timestamp(char *buf, size_t len)
{
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    strftime(buf, len, "%H:%M:%S", tm_info);
}

/* -------------------------------------------------------------------------- */
/* Find a client by file descriptor                                           */
/* -------------------------------------------------------------------------- */
static client_t *find_client(int fd)
{
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (clients[i].fd == fd)
            return &clients[i];
    }
    return NULL;
}

/* -------------------------------------------------------------------------- */
/* Find a client by nickname                                                  */
/* -------------------------------------------------------------------------- */
static client_t *find_client_by_nick(const char *nick)
{
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (clients[i].fd != -1 && clients[i].has_nick &&
            strcmp(clients[i].nickname, nick) == 0)
            return &clients[i];
    }
    return NULL;
}

/* -------------------------------------------------------------------------- */
/* Add a new pollfd entry                                                     */
/* -------------------------------------------------------------------------- */
static void add_to_pfds(int newfd)
{
    pfds[pfd_count].fd = newfd;
    pfds[pfd_count].events = POLLIN;
    pfd_count++;
}

/* -------------------------------------------------------------------------- */
/* Remove a pollfd entry by index                                             */
/* -------------------------------------------------------------------------- */
static void del_from_pfds(int idx)
{
    pfds[idx] = pfds[pfd_count - 1];
    pfd_count--;
}

/* -------------------------------------------------------------------------- */
/* Send a message to a single client                                          */
/* -------------------------------------------------------------------------- */
static void send_to_client(int fd, const char *msg)
{
    size_t len = strlen(msg);
    send(fd, msg, len, 0);
}

/* -------------------------------------------------------------------------- */
/* Broadcast a message to all clients except the sender                       */
/* -------------------------------------------------------------------------- */
static void broadcast(int sender_fd, const char *msg)
{
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (clients[i].fd != -1 && clients[i].fd != sender_fd &&
            clients[i].has_nick) {
            send_to_client(clients[i].fd, msg);
        }
    }
}

/* -------------------------------------------------------------------------- */
/* Handle the /who command — list all connected users                         */
/* -------------------------------------------------------------------------- */
static void handle_who(int fd)
{
    char buf[BUFFER_SIZE];
    int offset = 0;

    offset += snprintf(buf + offset, sizeof(buf) - offset,
                       "[Server] Connected users:\n");

    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (clients[i].fd != -1 && clients[i].has_nick) {
            offset += snprintf(buf + offset, sizeof(buf) - offset,
                               "  - %s\n", clients[i].nickname);
        }
    }
    send_to_client(fd, buf);
}

/* -------------------------------------------------------------------------- */
/* Handle the /msg command — send a private message                           */
/* -------------------------------------------------------------------------- */
static void handle_private_msg(int sender_fd, const char *args)
{
    char target_nick[NICK_SIZE];
    const char *space;
    client_t *sender;
    client_t *target;
    char buf[MSG_SIZE];
    char ts[16];

    /* Parse: /msg <nick> <message> */
    space = strchr(args, ' ');
    if (!space) {
        send_to_client(sender_fd,
                       "[Server] Usage: /msg <username> <message>\n");
        return;
    }

    size_t nick_len = (size_t)(space - args);
    if (nick_len >= NICK_SIZE) nick_len = NICK_SIZE - 1;
    strncpy(target_nick, args, nick_len);
    target_nick[nick_len] = '\0';

    sender = find_client(sender_fd);
    target = find_client_by_nick(target_nick);

    if (!target) {
        snprintf(buf, sizeof(buf), "[Server] User '%s' not found.\n",
                 target_nick);
        send_to_client(sender_fd, buf);
        return;
    }

    if (target->fd == sender_fd) {
        send_to_client(sender_fd,
                       "[Server] You can't message yourself.\n");
        return;
    }

    get_timestamp(ts, sizeof(ts));

    /* Send to target */
    snprintf(buf, sizeof(buf), "[%s] [PM from %s] %s\n",
             ts, sender->nickname, space + 1);
    send_to_client(target->fd, buf);

    /* Confirm to sender */
    snprintf(buf, sizeof(buf), "[%s] [PM to %s] %s\n",
             ts, target_nick, space + 1);
    send_to_client(sender_fd, buf);
}

/* -------------------------------------------------------------------------- */
/* Handle the /help command                                                   */
/* -------------------------------------------------------------------------- */
static void handle_help(int fd)
{
    const char *help =
        "[Server] Available commands:\n"
        "  /who   — List connected users\n"
        "  /msg <user> <text> — Send a private message\n"
        "  /help  — Show this help\n"
        "  /quit  — Disconnect from the server\n";
    send_to_client(fd, help);
}

/* -------------------------------------------------------------------------- */
/* Get the listener socket (from Beej's guide)                                */
/* -------------------------------------------------------------------------- */
static int get_listener_socket(const char *port)
{
    int listener;
    int yes = 1;
    int rv;
    struct addrinfo hints, *ai, *p;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_UNSPEC;    /* IPv4 or IPv6 */
    hints.ai_socktype = SOCK_STREAM;  /* TCP */
    hints.ai_flags    = AI_PASSIVE;   /* fill in my IP */

    if ((rv = getaddrinfo(NULL, port, &hints, &ai)) != 0) {
        fprintf(stderr, "server: getaddrinfo: %s\n", gai_strerror(rv));
        return -1;
    }

    for (p = ai; p != NULL; p = p->ai_next) {
        listener = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (listener < 0)
            continue;

        setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int));

        if (bind(listener, p->ai_addr, p->ai_addrlen) < 0) {
            close(listener);
            continue;
        }
        break;
    }

    freeaddrinfo(ai);

    if (p == NULL)
        return -1;

    if (listen(listener, 10) == -1)
        return -1;

    return listener;
}

/* -------------------------------------------------------------------------- */
/* Main                                                                       */
/* -------------------------------------------------------------------------- */
int main(int argc, char *argv[])
{
    const char *port = DEFAULT_PORT;
    int listener;
    char buf[BUFFER_SIZE];
    char msg[MSG_SIZE];
    char ts[16];

    if (argc >= 2)
        port = argv[1];

    /* Initialize client slots */
    for (int i = 0; i < MAX_CLIENTS; i++)
        clients[i].fd = -1;

    /* Set up signal handler */
    struct sigaction sa;
    sa.sa_handler = handle_sigint;
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);

    /* Create listener socket */
    listener = get_listener_socket(port);
    if (listener == -1) {
        fprintf(stderr, "Error: Could not create listener socket on port %s\n",
                port);
        exit(1);
    }

    /* Add listener to pollfd set */
    add_to_pfds(listener);

    printf("=== TCP Chat Server ===\n");
    printf("Listening on port %s\n", port);
    printf("Press Ctrl+C to shut down\n\n");

    /* Main event loop */
    while (running) {
        int poll_count = poll(pfds, pfd_count, 1000); /* 1s timeout */

        if (poll_count == -1) {
            if (errno == EINTR) continue; /* interrupted by signal */
            perror("poll");
            break;
        }

        for (int i = 0; i < pfd_count; i++) {
            if (!(pfds[i].revents & POLLIN))
                continue;

            if (pfds[i].fd == listener) {
                /* ---- New connection ---- */
                struct sockaddr_storage remoteaddr;
                socklen_t addrlen = sizeof(remoteaddr);
                int newfd = accept(listener,
                                   (struct sockaddr *)&remoteaddr,
                                   &addrlen);
                if (newfd == -1) {
                    perror("accept");
                    continue;
                }

                /* Find a free client slot */
                int slot = -1;
                for (int j = 0; j < MAX_CLIENTS; j++) {
                    if (clients[j].fd == -1) {
                        slot = j;
                        break;
                    }
                }

                if (slot == -1) {
                    send_to_client(newfd,
                                   "[Server] Server is full. Try again later.\n");
                    close(newfd);
                    continue;
                }

                /* Register the new client */
                clients[slot].fd = newfd;
                clients[slot].has_nick = 0;
                memset(clients[slot].nickname, 0, NICK_SIZE);

                add_to_pfds(newfd);

                /* Prompt for nickname */
                send_to_client(newfd,
                    "Welcome to TCP Chat! Enter your nickname: ");

                char addr_str[INET6_ADDRSTRLEN];
                if (remoteaddr.ss_family == AF_INET) {
                    inet_ntop(AF_INET,
                        &((struct sockaddr_in *)&remoteaddr)->sin_addr,
                        addr_str, sizeof(addr_str));
                } else {
                    inet_ntop(AF_INET6,
                        &((struct sockaddr_in6 *)&remoteaddr)->sin6_addr,
                        addr_str, sizeof(addr_str));
                }
                printf("[+] New connection from %s on socket %d\n",
                       addr_str, newfd);

            } else {
                /* ---- Data from existing client ---- */
                int nbytes = recv(pfds[i].fd, buf, sizeof(buf) - 1, 0);

                if (nbytes <= 0) {
                    /* Connection closed or error */
                    client_t *c = find_client(pfds[i].fd);
                    if (c) {
                        if (c->has_nick) {
                            get_timestamp(ts, sizeof(ts));
                            snprintf(msg, sizeof(msg),
                                     "[%s] *** %s has left the chat ***\n",
                                     ts, c->nickname);
                            printf("[-] %s disconnected (socket %d)\n",
                                   c->nickname, pfds[i].fd);
                            broadcast(pfds[i].fd, msg);
                        }
                        c->fd = -1;
                        c->has_nick = 0;
                    }
                    close(pfds[i].fd);
                    del_from_pfds(i);
                    i--; /* recheck this index */
                    continue;
                }

                buf[nbytes] = '\0';

                /* Strip trailing newline/carriage return */
                while (nbytes > 0 &&
                       (buf[nbytes - 1] == '\n' || buf[nbytes - 1] == '\r')) {
                    buf[--nbytes] = '\0';
                }

                if (nbytes == 0)
                    continue;

                client_t *c = find_client(pfds[i].fd);
                if (!c) continue;

                if (!c->has_nick) {
                    /* This is the nickname message */
                    /* Check for duplicate nickname */
                    if (find_client_by_nick(buf)) {
                        send_to_client(pfds[i].fd,
                            "[Server] Nickname already taken. "
                            "Choose another: ");
                        continue;
                    }

                    strncpy(c->nickname, buf, NICK_SIZE - 1);
                    c->nickname[NICK_SIZE - 1] = '\0';
                    c->has_nick = 1;

                    get_timestamp(ts, sizeof(ts));
                    printf("[+] Socket %d set nickname to '%s'\n",
                           pfds[i].fd, c->nickname);

                    /* Welcome the new user */
                    snprintf(msg, sizeof(msg),
                             "[%s] Welcome, %s! "
                             "Type /help for commands.\n",
                             ts, c->nickname);
                    send_to_client(pfds[i].fd, msg);

                    /* Announce to others */
                    snprintf(msg, sizeof(msg),
                             "[%s] *** %s has joined the chat ***\n",
                             ts, c->nickname);
                    broadcast(pfds[i].fd, msg);

                } else if (buf[0] == '/') {
                    /* Command handling */
                    if (strcmp(buf, "/quit") == 0) {
                        get_timestamp(ts, sizeof(ts));
                        snprintf(msg, sizeof(msg),
                                 "[%s] *** %s has left the chat ***\n",
                                 ts, c->nickname);
                        printf("[-] %s quit (socket %d)\n",
                               c->nickname, pfds[i].fd);
                        broadcast(pfds[i].fd, msg);
                        send_to_client(pfds[i].fd, "[Server] Goodbye!\n");
                        close(pfds[i].fd);
                        c->fd = -1;
                        c->has_nick = 0;
                        del_from_pfds(i);
                        i--;
                    } else if (strcmp(buf, "/who") == 0) {
                        handle_who(pfds[i].fd);
                    } else if (strncmp(buf, "/msg ", 5) == 0) {
                        handle_private_msg(pfds[i].fd, buf + 5);
                    } else if (strcmp(buf, "/help") == 0) {
                        handle_help(pfds[i].fd);
                    } else {
                        send_to_client(pfds[i].fd,
                            "[Server] Unknown command. "
                            "Type /help for available commands.\n");
                    }
                } else {
                    /* Regular chat message — broadcast to all */
                    get_timestamp(ts, sizeof(ts));
                    snprintf(msg, sizeof(msg), "[%s] %s: %s\n",
                             ts, c->nickname, buf);
                    printf("%s", msg);
                    broadcast(pfds[i].fd, msg);
                }
            }
        }
    }

    /* Clean shutdown */
    printf("\nShutting down server...\n");
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (clients[i].fd != -1) {
            send_to_client(clients[i].fd,
                           "[Server] Server shutting down. Goodbye!\n");
            close(clients[i].fd);
        }
    }
    close(listener);
    printf("Server stopped.\n");

    return 0;
}
