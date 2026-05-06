/*
 * =============================================================================
 * TCP Chat Client
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
 * This application uses TCP (Transmission Control Protocol). See server.c for
 * the full rationale. In summary: TCP's reliability, ordered delivery, and
 * connection-oriented design are essential for a text chat application where
 * every message must arrive intact and in order.
 *
 * =============================================================================
 * DESCRIPTION
 * =============================================================================
 * This client connects to the TCP chat server and provides an interactive
 * terminal interface for sending and receiving messages. It uses poll() to
 * simultaneously monitor:
 *   - stdin for user input
 *   - the server socket for incoming messages
 *
 * Supported commands (handled server-side):
 *   /who   — List connected users
 *   /msg <user> <text> — Send a private message
 *   /help  — Show available commands
 *   /quit  — Disconnect from the server
 *
 * Usage: ./client [hostname] [port]
 *        Defaults: hostname=127.0.0.1, port=9034
 *
 * Based on examples from Beej's Guide (Sections 6.2, 6.3).
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <poll.h>
#include <signal.h>

#define DEFAULT_HOST  "127.0.0.1"
#define DEFAULT_PORT  "9034"
#define BUFFER_SIZE   4096

static volatile sig_atomic_t running = 1;

/* -------------------------------------------------------------------------- */
/* Signal handler for clean exit                                              */
/* -------------------------------------------------------------------------- */
static void handle_sigint(int sig)
{
    (void)sig;
    running = 0;
}

/* -------------------------------------------------------------------------- */
/* Connect to the chat server (from Beej's guide)                             */
/* -------------------------------------------------------------------------- */
static int connect_to_server(const char *host, const char *port)
{
    int sockfd;
    int rv;
    struct addrinfo hints, *servinfo, *p;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_UNSPEC;    /* IPv4 or IPv6 */
    hints.ai_socktype = SOCK_STREAM;  /* TCP */

    if ((rv = getaddrinfo(host, port, &hints, &servinfo)) != 0) {
        fprintf(stderr, "client: getaddrinfo: %s\n", gai_strerror(rv));
        return -1;
    }

    for (p = servinfo; p != NULL; p = p->ai_next) {
        sockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (sockfd == -1)
            continue;

        if (connect(sockfd, p->ai_addr, p->ai_addrlen) == -1) {
            close(sockfd);
            continue;
        }
        break;
    }

    freeaddrinfo(servinfo);

    if (p == NULL) {
        fprintf(stderr, "client: failed to connect to %s:%s\n", host, port);
        return -1;
    }

    return sockfd;
}

/* -------------------------------------------------------------------------- */
/* Main                                                                       */
/* -------------------------------------------------------------------------- */
int main(int argc, char *argv[])
{
    const char *host = DEFAULT_HOST;
    const char *port = DEFAULT_PORT;
    int sockfd;
    char buf[BUFFER_SIZE];
    struct pollfd pfds[2];

    if (argc >= 2)
        host = argv[1];
    if (argc >= 3)
        port = argv[2];

    /* Set up signal handler */
    struct sigaction sa;
    sa.sa_handler = handle_sigint;
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);

    printf("Connecting to %s:%s...\n", host, port);

    sockfd = connect_to_server(host, port);
    if (sockfd == -1) {
        fprintf(stderr, "Could not connect to server.\n");
        exit(1);
    }

    printf("Connected! Waiting for server prompt...\n\n");

    /* Set up poll: monitor stdin (index 0) and server socket (index 1) */
    pfds[0].fd = STDIN_FILENO;
    pfds[0].events = POLLIN;
    pfds[1].fd = sockfd;
    pfds[1].events = POLLIN;

    while (running) {
        int poll_count = poll(pfds, 2, 1000);

        if (poll_count == -1) {
            if (errno == EINTR) continue;
            perror("poll");
            break;
        }

        /* Check for incoming data from server */
        if (pfds[1].revents & POLLIN) {
            int nbytes = recv(sockfd, buf, sizeof(buf) - 1, 0);
            if (nbytes <= 0) {
                if (nbytes == 0) {
                    printf("\nServer closed the connection.\n");
                } else {
                    perror("recv");
                }
                break;
            }
            buf[nbytes] = '\0';
            printf("%s", buf);
            fflush(stdout);
        }

        /* Check for user input from stdin */
        if (pfds[0].revents & POLLIN) {
            if (fgets(buf, sizeof(buf), stdin) == NULL) {
                /* EOF (Ctrl+D) */
                printf("\nDisconnecting...\n");
                send(sockfd, "/quit\n", 6, 0);
                break;
            }

            /* Check if user wants to quit locally */
            size_t len = strlen(buf);
            /* Strip trailing newline for comparison */
            char cmd[BUFFER_SIZE];
            strncpy(cmd, buf, sizeof(cmd));
            cmd[sizeof(cmd) - 1] = '\0';
            while (len > 0 && (cmd[len - 1] == '\n' || cmd[len - 1] == '\r'))
                cmd[--len] = '\0';

            if (strcmp(cmd, "/quit") == 0) {
                send(sockfd, buf, strlen(buf), 0);
                printf("Disconnecting...\n");
                break;
            }

            /* Send the message to the server */
            if (send(sockfd, buf, strlen(buf), 0) == -1) {
                perror("send");
                break;
            }
        }
    }

    close(sockfd);
    printf("Disconnected.\n");

    return 0;
}
