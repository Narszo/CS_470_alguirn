# TCP Multi-Client Chat Application
**CS 470 - Final Assignment: Network Socket Programming**  
**Author: Noah Alguire**  
**Spring 2026**

## Overview
A TCP-based multi-client chat application written in C using POSIX sockets. The server handles multiple simultaneous clients using `poll()` for I/O multiplexing. Communication runs over `127.0.0.1` (localhost) on a configurable port.

**Reference:** [Beej's Guide to Network Programming](https://beej.us/guide/bgnet/)

## Why TCP?
TCP was chosen over UDP because:
- **Reliability:** Every chat message is guaranteed to be delivered.
- **Ordered delivery:** Messages arrive in the order they were sent.
- **Connection tracking:** TCP's connection model lets the server know exactly who is connected.
- **Flow control:** Prevents fast senders from overwhelming slow receivers.

UDP would be better suited for real-time applications (voice, video, games) where speed is prioritized over reliability.

## Features
- **Multi-client support** — up to 64 simultaneous connections
- **Nickname system** — users choose a display name on connect
- **Message broadcasting** — messages are relayed to all connected clients
- **Private messaging** — `/msg <user> <text>` sends a direct message
- **User listing** — `/who` shows all connected users
- **Join/leave notifications** — all users are notified when someone connects or disconnects
- **Timestamped messages** — all messages include `[HH:MM:SS]` timestamps
- **Graceful shutdown** — `Ctrl+C` cleanly terminates server and client

## Building

```bash
cd final_project
make
```

This produces two executables: `server` and `client`.

To clean:
```bash
make clean
```

## Running

### Start the Server
```bash
./server [port]
```
Default port is `9034`.

### Connect a Client
In a separate terminal:
```bash
./client [hostname] [port]
```
Defaults: hostname=`127.0.0.1`, port=`9034`.

### Example Session
```
# Terminal 1 — Start the server
./server

# Terminal 2 — Connect first client
./client
> Enter nickname: Alice

# Terminal 3 — Connect second client
./client
> Enter nickname: Bob

# Now Alice and Bob can chat!
Alice: Hello Bob!
Bob: Hi Alice! How are you?
```

## Commands
| Command | Description |
|---------|-------------|
| `/who` | List all connected users |
| `/msg <user> <text>` | Send a private message |
| `/help` | Show available commands |
| `/quit` | Disconnect from the server |

## Architecture
```
┌─────────┐     TCP      ┌──────────────┐     TCP      ┌─────────┐
│ Client 1 │◄────────────►│              │◄────────────►│ Client 2 │
└─────────┘              │   Server     │              └─────────┘
                         │  (poll()     │
┌─────────┐     TCP      │   event      │     TCP      ┌─────────┐
│ Client 3 │◄────────────►│   loop)      │◄────────────►│ Client N │
└─────────┘              └──────────────┘              └─────────┘
```

The server uses a single-threaded event loop with `poll()` to multiplex I/O across all connected sockets. This avoids the complexity of multi-threading while efficiently handling many clients.
