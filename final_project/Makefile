# =============================================================================
# Makefile for TCP Chat Application
# CS 470 - Final Assignment: Network Socket Programming
# =============================================================================

CC      = gcc
CFLAGS  = -Wall -Wextra -pedantic -std=c11 -D_POSIX_C_SOURCE=200809L -g
LDFLAGS =

# Targets
SERVER  = server
CLIENT  = client

.PHONY: all clean

all: $(SERVER) $(CLIENT)

$(SERVER): server.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(CLIENT): client.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(SERVER) $(CLIENT)
