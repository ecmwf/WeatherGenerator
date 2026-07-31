# WeatherGenerator internal dashboard


## Deployment

See full instruction at https://gitlab.jsc.fz-juelich.de/esde/WeatherGenerator-private/-/wikis/home/Tracking-progress

### Local run

```sh
uv run --env-file=.env streamlit run dashboard.py
```

The checked-in Streamlit config binds the dashboard to `127.0.0.1:8501` so it is not exposed directly. Use nginx as a reverse proxy to serve the dashboard on the standard web ports.

### HTTPS deployment with nginx and a self-signed certificate

This setup:

- serves HTTPS on port `443`;
- redirects HTTP port `80` to HTTPS;
- proxies requests to Streamlit on `127.0.0.1:8501`;
- uses a self-signed certificate generated on the deployment host.

Install nginx if it is not already installed, then generate a local certificate. Use the public DNS name or public IP address that users will type in the browser:

```sh
cd packages/dashboard
./deploy/generate_self_signed_cert.sh <public-dns-name-or-ip>
```

Install/reload the nginx configuration:

```sh
./deploy/install_nginx_config.sh
```

Then run Streamlit:

```sh
uv run --env-file=.env streamlit run dashboard.py
```

Open:

```text
https://<public-dns-name-or-ip>/
```

Browsers will show a warning because the certificate is self-signed. That is expected for this setup.

If you want nginx to use a specific host name in `server_name`, set `SERVER_NAME` when installing the config:

```sh
SERVER_NAME=<public-dns-name> ./deploy/install_nginx_config.sh
```

The generated private key is written to `deploy/certs/dashboard.key` and is ignored by git. Do not commit it.