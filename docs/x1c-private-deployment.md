# X1C Private Deployment

This guide deploys the complete AgentWeb RAG stack on the X1C Ubuntu host before any public
relay is configured. PostgreSQL, Milvus, MinIO, etcd, the API, the prediction runtime, and the
memory worker stay on the X1C. Only the Web and API diagnostic ports bind to the X1C loopback
interface, so they are not reachable from the home LAN or the Internet.

## Preconditions

The X1C host must already have Docker Engine and Docker Compose available, with the domestic
Docker mirror configured. `scripts/bootstrap-x1c-rag.sh` performs that one-time setup.

The application source must be present on the X1C, normally at:

```text
/home/yuanx/agentweb-rag
```

The deployment has been sized for the X1C (16 GiB RAM), not for the 1 GiB public relay. It runs
one API worker, one model instance at a time, two PyTorch compute threads, and an ONNX embedding
worker with a batch size of four. The API image uses Python 3.12 and exactly
`torch==2.6.0+cpu` from the Aliyun PyTorch wheel mirror. It must not install generic Linux
PyTorch or any NVIDIA/CUDA packages.

## First Private Deployment

On the X1C:

```bash
cd /home/yuanx/agentweb-rag
cp infra/compose/.env.x1c.example infra/compose/.env.x1c
chmod 600 infra/compose/.env.x1c
```

Edit `infra/compose/.env.x1c`. Replace the PostgreSQL `CHANGE_ME` value with a long random
password. Keep the documented MinIO compatibility values unchanged: this MinIO service is only
reachable on the internal Docker `rag-data` network and has no host port. Configure
`LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` for normal answering. The empty defaults are
suitable only for confirming that containers and model runtimes start.

Then build and launch the private stack:

```bash
./scripts/deploy-x1c-rag.sh
```

The `rag-bootstrap` one-shot service initializes PostgreSQL and the LangGraph checkpointer. When
the derived text-only retrieval input is absent from a clean release checkout, it deterministically
regenerates it from the versioned `rawData.jsonl` source before inspecting
`growth_records_minilm`. It imports the 6,292 source records only when the collection is missing,
empty, incomplete, or has a vector schema other than the fixed 384-dimensional MiniLM contract.
A healthy unchanged collection is not embedded again.

Check the stack at any time:

```bash
./scripts/check-x1c-rag.sh
docker compose --env-file infra/compose/.env.x1c -f infra/compose/docker-compose.x1c.yml ps
```

## Private Browser Validation

Keep the X1C ports private and access them from the Mac through SSH forwarding:

```bash
ssh -N \
  -L 3003:127.0.0.1:3003 \
  -L 8003:127.0.0.1:8003 \
  yuanx@192.168.1.18
```

Open `http://127.0.0.1:3003/login` on the Mac. The frontend proxies browser `/api/*` requests to
the internal `rag-api` container, so the 8003 forward is only useful for direct diagnostics such
as `http://127.0.0.1:8003/api/rag/health`.

Verify all of the following before enabling host autostart or configuring the public relay:

1. First login registers one account; a wrong password for that same email is rejected.
2. A sufficient material query returns literature evidence and does not invoke the predictor.
3. An unavailable/insufficient exact material query invokes the local prediction fallback and is
   visibly marked as an unverified candidate route.
4. Two accounts cannot read one another's sessions, long-term memories, or prediction history.
5. `http://127.0.0.1:8003/api/rag/health` reports `memory_database=postgres` and
   `short_term_backend=postgres-checkpointer`.

## Boot And Autostart

Enable autostart only after the private checks succeed:

```bash
cd /home/yuanx/agentweb-rag
sudo ./scripts/install-x1c-systemd.sh
```

The installer enables two independent, unattended services:

- `agentweb-rag.service` starts the existing Docker images and volumes. It deliberately does not
  rebuild images or download models at boot.
- `agentweb-rag-tunnel.service` waits for `network-online.target`, then opens a reverse SSH
  tunnel to the public relay. It publishes the web and API ports only on the relay loopback
  address (`127.0.0.1:13003` and `127.0.0.1:18003`). Nginx is the only public-facing process.

The tunnel uses SSH batch mode and a dedicated unencrypted private key at
`~/.ssh/id_ed25519_agentweb_tunnel`; an unattended system service cannot enter a passphrase.
Keep that private key owned by the deployment user with mode `0600`, and keep the relay host key
in `~/.ssh/known_hosts`. `Restart=always` with `StartLimitIntervalSec=0` makes the tunnel keep
retrying after a router, DNS, or relay outage rather than requiring an SSH login to recover.

The Docker stack and tunnel do **not** depend on a graphical desktop login. On this X1C, Ubuntu's
GDM configuration already enables automatic login for `yuanx` for local desktop convenience:

```ini
[daemon]
AutomaticLoginEnable=true
AutomaticLogin=yuanx
```

Keep `graphical.target`, `NetworkManager.service`, and `NetworkManager-wait-online.service`
enabled. These ensure that the desktop starts automatically, Wi-Fi is given time to associate,
and the tunnel starts only after the network has been declared online. After a source update, run
`./scripts/deploy-x1c-rag.sh` manually, check it, then the systemd unit will use the refreshed
images on later restarts.

Inspect an unattended boot or reconnect with:

```bash
sudo systemctl status agentweb-rag.service agentweb-rag-tunnel.service --no-pager
sudo journalctl -u agentweb-rag-tunnel.service -b --no-pager
```

## Network Boundary

The Compose file creates two Docker networks:

```text
rag-data (internal): PostgreSQL, Milvus, MinIO, etcd, API, worker, bootstrap
rag-edge:            API and Web only, used for the Web server's internal /api proxy
```

PostgreSQL, Milvus, MinIO, and etcd publish no host ports. The only host bindings are:

```text
127.0.0.1:3003 -> rag-web
127.0.0.1:8003 -> rag-api
```

The constrained public server must not be configured until this private deployment is stable. Its
future role is limited to HTTPS termination and a reverse tunnel terminating at the X1C loopback
ports; it must not receive database volumes, models, raw data, or API service credentials.

## Recovery Operations

View logs:

```bash
docker compose --env-file infra/compose/.env.x1c -f infra/compose/docker-compose.x1c.yml logs -f rag-bootstrap rag-api rag-web
```

Recreate Milvus deliberately after a source corpus or embedding-contract change. This is the one
operation that performs a full re-embedding:

```bash
RAG_RECREATE_MILVUS=1 docker compose \
  --env-file infra/compose/.env.x1c \
  -f infra/compose/docker-compose.x1c.yml \
  run --rm rag-bootstrap
```

Do not delete Docker named volumes as a troubleshooting shortcut. That would erase accounts,
sessions, long-term memory, prediction-run history, or the imported Milvus collection.
