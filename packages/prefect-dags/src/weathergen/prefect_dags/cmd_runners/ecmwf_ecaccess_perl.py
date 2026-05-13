"""
Thin Python client for the ECMWF ECaccess SOAP API (Apache Axis2).

Replaces the Perl-based ECaccess Web Toolkit for Slurm job management.
Authenticates using a pre-created .eccert.crt X509 certificate (valid 7 days).

"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import sys
import textwrap
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — taken directly from ECaccess.pm
# ---------------------------------------------------------------------------

# SOAP namespace: ECaccess.pm line 118
#   SOAP::Lite->uri('http://service.client.ecmwf/xsd')
SOAP_NS = "http://service.client.ecmwf/xsd"

SOAP_ENV_NS = "http://schemas.xmlsoap.org/soap/envelope/"
XSD_NS = "http://www.w3.org/2001/XMLSchema"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"

# Axis2 service path: ECaccess.pm line 118
AXIS2_PATH = "/axis2/services/ECaccessService"

# REST data I/O path: ECaccess.pm lines 274, 293
DATAIOS_PATH = "/dataios"

# Default gateway: ECaccess.pm lines 113-114
DEFAULT_GATEWAY = "boaccess.ecmwf.int"

# Default certificate location: ECaccess.pm lines 168-176
DEFAULT_CERT = Path.home() / ".eccert.crt"


# ---------------------------------------------------------------------------
# SOAP envelope builder
# ---------------------------------------------------------------------------


def _soap_envelope(method: str, params: list[tuple[str, str, str | None]]) -> bytes:
    """
    Build a SOAP 1.1 envelope for Axis2.

    params is a list of (name, value, xsi_type_or_None).
    """
    env = ET.Element(
        "soapenv:Envelope",
        {
            "xmlns:soapenv": SOAP_ENV_NS,
            "xmlns:xsd": XSD_NS,
            "xmlns:xsi": XSI_NS,
            "xmlns:ns": SOAP_NS,
        },
    )
    body = ET.SubElement(env, "soapenv:Body")
    meth = ET.SubElement(body, f"ns:{method}")

    for name, value, xsi_type in params:
        el = ET.SubElement(meth, f"ns:{name}")
        if xsi_type:
            el.set("xsi:type", xsi_type)
        el.text = value

    return ET.tostring(env, encoding="unicode").encode("utf-8")


def _soap_envelope_raw_inner(method: str, token_xml: str, inner_xml: str) -> bytes:
    """Build a SOAP envelope with raw XML inner content (for complex types)."""
    xml_str = textwrap.dedent(f"""\
        <soapenv:Envelope
            xmlns:soapenv="{SOAP_ENV_NS}"
            xmlns:xsd="{XSD_NS}"
            xmlns:xsi="{XSI_NS}"
            xmlns:ns="{SOAP_NS}">
          <soapenv:Body>
            <ns:{method}>
              {token_xml}
              {inner_xml}
            </ns:{method}>
          </soapenv:Body>
        </soapenv:Envelope>""")
    return xml_str.encode("utf-8")


# ---------------------------------------------------------------------------
# SOAP response parser
# ---------------------------------------------------------------------------


def _parse_fault(root: ET.Element) -> str | None:
    """Extract faultstring from a SOAP Fault, or None."""
    for elem in root.iter():
        local = elem.tag.rsplit("}", 1)[-1]
        if local == "Fault":
            for child in elem.iter():
                ctag = child.tag.rsplit("}", 1)[-1]
                if ctag == "faultstring" and child.text:
                    return child.text
    return None


def _find_response(root: ET.Element, method: str) -> ET.Element | None:
    """Find the <methodResponse> element."""
    target = f"{method}Response"
    for elem in root.iter():
        local = elem.tag.rsplit("}", 1)[-1]
        if local == target:
            return elem
    return None


def _get_returns(resp: ET.Element) -> list[ET.Element]:
    """Get all <return> child elements from a response."""
    results = []
    for elem in resp:
        local = elem.tag.rsplit("}", 1)[-1]
        if local == "return":
            results.append(elem)
    return results


def _return_text(resp: ET.Element) -> str:
    """Get the text of the first <return> element."""
    for elem in resp:
        local = elem.tag.rsplit("}", 1)[-1]
        if local == "return" and elem.text:
            return elem.text
    return ""


def _elem_field(elem: ET.Element, field: str) -> str:
    """Get text of a named child element."""
    for child in elem:
        local = child.tag.rsplit("}", 1)[-1]
        if local == field:
            return child.text or ""
    return ""


def _elem_fields(elem: ET.Element, field: str) -> list[str]:
    """Get text of all children with a given name."""
    results = []
    for child in elem:
        local = child.tag.rsplit("}", 1)[-1]
        if local == field and child.text:
            results.append(child.text)
    return results


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class ECaccessClient:
    """Minimal ECaccess SOAP client for Slurm job management."""

    def __init__(
        self,
        cert_path: Path,
        https_host: str = DEFAULT_GATEWAY,
        http_host: str = DEFAULT_GATEWAY,
        debug: bool = False,
    ):
        self.cert_path = cert_path
        self.https_host = https_host
        self.http_host = http_host
        self.debug = debug

        # Control channel: HTTPS (ECaccess.pm line 117-118)
        self._ctrl_url = f"https://{https_host}{AXIS2_PATH}"
        # Data channel: HTTP (ECaccess.pm line 122-123)
        self._data_url = f"http://{http_host}{AXIS2_PATH}"
        # REST data I/O (ECaccess.pm lines 271, 290)
        self._dataios_url = f"http://{http_host}{DATAIOS_PATH}"

        self._token: str | None = None

        # TLS verification disabled — matching ECaccess.pm line 33:
        #   IO::Socket::SSL::set_ctx_defaults(SSL_verify_mode => 0)
        self._https = httpx.Client(timeout=120.0, verify=False, follow_redirects=True)
        self._http = httpx.Client(timeout=120.0, follow_redirects=True)

    # -- low-level SOAP call ------------------------------------------------

    def _call(
        self, method: str, params: list[tuple[str, str, str | None]], *, use_https: bool = True
    ) -> ET.Element:
        """Make a SOAP call and return the parsed response element."""
        envelope = _soap_envelope(method, params)
        url = self._ctrl_url if use_https else self._data_url
        client = self._https if use_https else self._http

        if self.debug:
            logger.debug(">>> POST %s", url)
            logger.debug("    SOAPAction: urn:%s", method)
            logger.debug("%s", envelope.decode())

        resp = client.post(
            url,
            content=envelope,
            headers={
                "Content-Type": "text/xml; charset=utf-8",
                "SOAPAction": f"urn:{method}",
            },
        )

        if self.debug:
            logger.debug("<<< HTTP %s", resp.status_code)
            logger.debug("%s", resp.text[:2000])

        root = ET.fromstring(resp.content)

        fault = _parse_fault(root)
        if fault:
            raise RuntimeError(f"SOAP Fault: {fault}")

        response = _find_response(root, method)
        if response is None:
            # Some methods return empty body on success (e.g. deleteJob)
            if resp.status_code < 400:
                return ET.Element("empty")
            raise RuntimeError(f"No {method}Response in reply (HTTP {resp.status_code})")

        return response

    def _call_raw(self, method: str, envelope: bytes) -> ET.Element:
        """Make a SOAP call with a pre-built envelope."""
        if self.debug:
            logger.debug(">>> POST %s", self._ctrl_url)
            logger.debug("%s", envelope.decode()[:2000])

        resp = self._https.post(
            self._ctrl_url,
            content=envelope,
            headers={
                "Content-Type": "text/xml; charset=utf-8",
                "SOAPAction": f"urn:{method}",
            },
        )

        if self.debug:
            logger.debug("<<< HTTP %s", resp.status_code)
            logger.debug("%s", resp.text[:2000])

        root = ET.fromstring(resp.content)
        fault = _parse_fault(root)
        if fault:
            raise RuntimeError(f"SOAP Fault: {fault}")

        response = _find_response(root, method)
        if response is None:
            if resp.status_code < 400:
                return ET.Element("empty")
            raise RuntimeError(f"No {method}Response in reply (HTTP {resp.status_code})")
        return response

    # -- authentication (ECaccess.pm lines 185-237) -------------------------

    def authenticate(self) -> str:
        if self._token:
            return self._token

        if not self.cert_path.exists():
            raise FileNotFoundError(
                f"Certificate not found: {self.cert_path}\n"
                "Create one via the web interface at https://boaccess.ecmwf.int/\n"
                "or with:  ecaccess-certificate-create"
            )

        # ECaccess.pm line 236:
        #   encode_base64($certificateContent)
        raw = self.cert_path.read_bytes()
        cert_b64 = base64.b64encode(raw).decode("ascii")

        resp = self._call(
            "getTokenFromCertificate",
            [
                ("certificate", cert_b64, None),
            ],
        )
        self._token = _return_text(resp)
        if not self._token:
            raise RuntimeError("Empty token — certificate may be expired or invalid")
        return self._token

    def release_token(self) -> None:
        if self._token:
            try:
                self._call("releaseToken", [("token", self._token, None)])
            except Exception:
                pass
            self._token = None

    # -- REST data I/O (ECaccess.pm lines 267-307) --------------------------

    def _upload_data(self, handle: str, data: bytes) -> None:
        """Upload data via REST POST to /dataios (multipart/form-data).

        Matches ECaccess.pm writeFileOutputStream (lines 285-307).
        """
        boundary = "---------------------------154328737501"
        body = (
            (
                f"-----------------------------154328737501\r\n"
                f'Content-Disposition: form-data; name="fileupload"; '
                f'filename="{urllib.parse.quote(handle)}"\r\n'
                f"Content-Type: application/octet-stream\r\n\r\n"
            ).encode()
            + data
            + b"\r\n-----------------------------154328737501--\r\n"
        )

        resp = self._http.post(
            self._dataios_url,
            content=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "python-ecaccess:0.1",
            },
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Upload failed: HTTP {resp.status_code}")

    def _download_data(self, handle: str) -> bytes:
        """Download data via REST GET from /dataios?handle=<handle>.

        Matches ECaccess.pm getFileInputStream (lines 267-281).
        """
        url = f"{self._dataios_url}?handle={urllib.parse.quote(handle)}"
        resp = self._http.get(
            url,
            headers={
                "User-Agent": "python-ecaccess:0.1",
                "Content-Type": "text/xml",
            },
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Download failed: HTTP {resp.status_code}")
        return resp.content

    # -- helper: close handle -----------------------------------------------

    def _close_handle(self, handle: str) -> None:
        self._call("closeHandle", [("handle", handle, None)])

    # -- gateway (no auth needed) -------------------------------------------

    def get_gateway_name(self) -> str:
        resp = self._call("getGatewayName", [])
        return _return_text(resp)

    # -- certificate operations list ----------------------------------------

    def list_operations(self) -> list[dict[str, str]]:
        """ecaccess-certificate-list: list available operations."""
        token = self.authenticate()
        resp = self._call("getOperationList", [("token", token, None)])
        results = []
        for ret in _get_returns(resp):
            results.append(
                {
                    "name": _elem_field(ret, "name"),
                    "duration": _elem_field(ret, "duration"),
                    "endDate": _elem_field(ret, "endDate"),
                    "comment": _elem_field(ret, "comment"),
                }
            )
        return results

    # -- queue list ----------------------------------------------------------

    def list_queues(self, detail_queue: str | None = None) -> list[dict[str, str]]:
        """ecaccess-queue-list."""
        token = self.authenticate()
        if detail_queue:
            resp = self._call(
                "getQueueDetail",
                [
                    ("token", token, None),
                    ("queueName", detail_queue, None),
                ],
            )
            results = []
            for ret in _get_returns(resp):
                results.append(
                    {
                        "name": _elem_field(ret, "name"),
                        "comment": _elem_field(ret, "comment"),
                    }
                )
            return results

        resp = self._call("getQueueList", [("token", token, None)])
        results = []
        for ret in _get_returns(resp):
            results.append(
                {
                    "queueName": _elem_field(ret, "queueName"),
                    "schedulerName": _elem_field(ret, "schedulerName"),
                    "comment": _elem_field(ret, "comment"),
                    "INIT": _elem_field(ret, "numberOfJobsInInitState"),
                    "WAIT": _elem_field(ret, "numberOfJobsInWaitState"),
                    "EXEC": _elem_field(ret, "numberOfJobsInExecState"),
                    "DONE": _elem_field(ret, "numberOfJobsInDoneState"),
                    "STOP": _elem_field(ret, "numberOfJobsInStopState"),
                }
            )
        return results

    # -- job list / detail ---------------------------------------------------

    def list_jobs(self, job_id: str | None = None) -> list[dict[str, str | list[str]]]:
        """ecaccess-job-list."""
        token = self.authenticate()

        if job_id:
            resp = self._call(
                "getJob",
                [
                    ("token", token, None),
                    ("jobid", job_id, None),
                ],
            )
            ret = _get_returns(resp)
            if not ret:
                return []
            job = ret[0]
            return [
                {
                    "jobId": _elem_field(job, "jobId"),
                    "name": _elem_field(job, "name"),
                    "queueName": _elem_field(job, "queueName"),
                    "hostName": _elem_field(job, "hostName"),
                    "scheduledDate": _elem_field(job, "scheduledDate"),
                    "expirationDate": _elem_field(job, "expirationDate"),
                    "tryDone": _elem_field(job, "tryDone"),
                    "tryCount": _elem_field(job, "tryCount"),
                    "status": _elem_field(job, "status"),
                    "comment": _elem_field(job, "comment"),
                    "outputFileSize": _elem_field(job, "outputFileSize"),
                    "errorFileSize": _elem_field(job, "errorFileSize"),
                    "inputFileSize": _elem_field(job, "inputFileSize"),
                    "eventIds": _elem_fields(job, "eventIds"),
                }
            ]

        resp = self._call("getJobList", [("token", token, None)])
        results = []
        for ret in _get_returns(resp):
            results.append(
                {
                    "jobId": _elem_field(ret, "jobId"),
                    "queueName": _elem_field(ret, "queueName"),
                    "status": _elem_field(ret, "status"),
                    "tryDone": _elem_field(ret, "tryDone"),
                    "tryCount": _elem_field(ret, "tryCount"),
                    "scheduledDate": _elem_field(ret, "scheduledDate"),
                    "name": _elem_field(ret, "name"),
                    "eventIds": _elem_fields(ret, "eventIds"),
                }
            )
        return results

    # -- job submit (ecaccess-job-submit) ------------------------------------

    def submit_job(
        self,
        script_path: str,
        queue: str | None = None,
        job_name: str | None = None,
        distant: bool = False,
        event_ids: str | None = None,
        scheduled_date: str | None = None,
        no_directives: bool = False,
        stderr_to_stdout: bool = False,
        life_time: int = 7,
        retry_count: int = 0,
        retry_frequency: int = 600,
    ) -> str:
        """Submit a job. Returns the ECaccess job ID.

        Matches ecaccess-job-submit Perl script flow:
        1. getTemporaryFile -> temp path
        2. Upload local script to temp path via handle + REST
           (or copyFile if --distant)
        3. submitJob with nested JobRequest
        4. deleteFile temp
        """
        token = self.authenticate()

        # Step 1: get a temporary file at ECMWF
        resp = self._call("getTemporaryFile", [("token", token, None)])
        temp_file = _return_text(resp)
        if not temp_file:
            raise RuntimeError("Failed to get temporary file")

        # Step 2: upload the script
        if not distant:
            # Local file — upload via handle + REST
            path = Path(script_path)
            if not path.exists():
                raise FileNotFoundError(f"Script not found: {script_path}")
            script_data = path.read_bytes()
            name = job_name or path.name

            # Get output handle
            resp = self._call(
                "getOutputFileHandle",
                [
                    ("token", token, None),
                    ("target", temp_file, None),
                    ("offset", "0", None),
                    ("umask", "640", None),
                ],
            )
            handle = _return_text(resp)
            if not handle:
                raise RuntimeError("Failed to get output file handle")

            # Upload via REST
            self._upload_data(handle, script_data)

            # Close handle
            self._close_handle(handle)
        else:
            # Remote file — copy to temp
            name = job_name or script_path.rsplit("/", 1)[-1].rsplit(":", 1)[-1]
            self._call(
                "copyFile",
                [
                    ("token", token, None),
                    ("source", script_path, None),
                    ("target", temp_file, None),
                    ("erase", "false", "xsd:boolean"),
                ],
            )

        # Step 3: submitJob with nested request
        def _bool(v: bool) -> str:
            return "true" if v else "false"

        def _opt(tag: str, val: str | None) -> str:
            if val is None:
                return f'<ns:{tag} xsi:nil="true"/>'
            safe = val.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            return f"<ns:{tag}>{safe}</ns:{tag}>"

        request_fields = [
            _opt("scheduledDate", scheduled_date),
            _opt("userMailAddress", None),
            '<ns:sendMailOnStart xsi:type="xsd:boolean">false</ns:sendMailOnStart>',
            '<ns:sendMailOnSuccess xsi:type="xsd:boolean">false</ns:sendMailOnSuccess>',
            '<ns:sendMailOnFailure xsi:type="xsd:boolean">false</ns:sendMailOnFailure>',
            '<ns:sendMailOnRetry xsi:type="xsd:boolean">false</ns:sendMailOnRetry>',
            f'<ns:containsDirectives xsi:type="xsd:boolean">'
            f"{_bool(not no_directives)}</ns:containsDirectives>",
            _opt("queueName", queue),
            _opt("name", name),
            _opt("transferGatewayName", None),
            _opt("transferRemoteLocation", None),
            '<ns:transferOutputFile xsi:type="xsd:boolean">false</ns:transferOutputFile>',
            '<ns:transferErrorFile xsi:type="xsd:boolean">false</ns:transferErrorFile>',
            '<ns:transferInputFile xsi:type="xsd:boolean">false</ns:transferInputFile>',
            '<ns:transferKeepInSpool xsi:type="xsd:boolean">false</ns:transferKeepInSpool>',
            '<ns:renewSubscription xsi:type="xsd:boolean">true</ns:renewSubscription>',
            f'<ns:errorToOutput xsi:type="xsd:boolean">'
            f"{_bool(stderr_to_stdout)}</ns:errorToOutput>",
            _opt("manPageContent", None),
            f"<ns:lifeTime>{life_time}</ns:lifeTime>",
            f"<ns:retryCount>{retry_count}</ns:retryCount>",
            f"<ns:retryFrequency>{retry_frequency}</ns:retryFrequency>",
            _opt("eventIds", event_ids),
            f"<ns:inputFile>{temp_file}</ns:inputFile>",
        ]

        token_xml = f"<ns:token>{token}</ns:token>"
        request_xml = (
            "<ns:request>\n"
            + "\n".join(f"        {f}" for f in request_fields)
            + "\n      </ns:request>"
        )
        envelope = _soap_envelope_raw_inner("submitJob", token_xml, request_xml)

        resp = self._call_raw("submitJob", envelope)
        job_id = _return_text(resp)

        # Step 4: cleanup temp file
        try:
            self._call(
                "deleteFile",
                [
                    ("token", token, None),
                    ("source", temp_file, None),
                    ("force", "true", "xsd:boolean"),
                ],
            )
        except Exception:
            pass

        return job_id

    # -- job delete ----------------------------------------------------------

    def delete_job(self, job_id: str) -> None:
        """ecaccess-job-delete."""
        token = self.authenticate()
        self._call(
            "deleteJob",
            [
                ("token", token, None),
                ("jobid", job_id, None),
            ],
        )

    # -- job restart ---------------------------------------------------------

    def restart_job(self, job_id: str) -> None:
        """ecaccess-job-restart."""
        token = self.authenticate()
        self._call(
            "restartJob",
            [
                ("token", token, None),
                ("jobId", job_id, None),
            ],
        )

    # -- job get output/error/input ------------------------------------------

    def get_job_output(self, job_id: str, which: str = "output") -> bytes:
        """ecaccess-job-get: download job output/error/input via REST.

        Matches Perl flow:
        1. getJobOutputHandle/ErrorHandle/InputHandle -> handle
        2. GET /dataios?handle=<handle> -> data
        3. closeHandle
        """
        token = self.authenticate()

        method = {
            "output": "getJobOutputHandle",
            "error": "getJobErrorHandle",
            "input": "getJobInputHandle",
        }[which]

        resp = self._call(
            method,
            [
                ("token", token, None),
                ("jobid", job_id, None),
            ],
        )
        handle = _return_text(resp)
        if not handle:
            raise RuntimeError(f"No handle returned for job {job_id} {which}")

        try:
            data = self._download_data(handle)
        finally:
            try:
                self._close_handle(handle)
            except Exception:
                pass

        return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_ops(client: ECaccessClient) -> None:
    ops = client.list_operations()
    if not ops:
        logger.info("(no operations — token may be invalid)")
        return
    for op in ops:
        logger.info("%-20s %-8s %-20s %s", op["name"], op["duration"], op["endDate"], op["comment"])


def _print_queues(client: ECaccessClient, detail: str | None) -> None:
    queues = client.list_queues(detail_queue=detail)
    if not queues:
        logger.info("(no queues)")
        return
    if detail:
        for q in queues:
            logger.info("%-20s %s", q["name"], q["comment"])
    else:
        for q in queues:
            counts = (
                f"INIT={q['INIT']},WAIT={q['WAIT']},EXEC={q['EXEC']},"
                f"DONE={q['DONE']},STOP={q['STOP']}"
            )
            logger.info(
                "%-15s %-15s %s (%s)",
                q["queueName"],
                q["schedulerName"],
                q["comment"],
                counts,
            )


def _print_jobs(client: ECaccessClient, job_id: str | None) -> None:
    jobs = client.list_jobs(job_id=job_id)
    if not jobs:
        logger.info("(no jobs)")
        return
    if job_id:
        j = jobs[0]
        logger.info("     Job-Id: %s", j["jobId"])
        if j["name"]:
            logger.info("   Job Name: %s", j["name"])
        logger.info("      Queue: %s", j["queueName"])
        if j["hostName"]:
            logger.info("       Host: %s", j["hostName"])
        logger.info("   Schedule: %s", j["scheduledDate"])
        logger.info(" Expiration: %s", j["expirationDate"])
        logger.info("  Try Count: %s/%s", j["tryDone"], j["tryCount"])
        logger.info("     Status: %s", j["status"])
        if j["eventIds"]:
            logger.info("  Event-Ids: %s", ";".join(j["eventIds"]))
        if j["status"] == "DONE":
            if j["outputFileSize"] and j["outputFileSize"] != "-1":
                logger.info("Stdout Size: %s", j["outputFileSize"])
            if j["errorFileSize"] and j["errorFileSize"] != "-1":
                logger.info("Stderr Size: %s", j["errorFileSize"])
        if j["comment"]:
            logger.info("    Comment: %s", j["comment"])
    else:
        for j in jobs:
            events = "[" + ";".join(j["eventIds"]) + "]" if j["eventIds"] else "[-]"
            tries = f"{j['tryDone']}/{j['tryCount']}"
            name = f" {j['name']}" if j["name"] else ""
            logger.info(
                "%-10s %-10s %-10s %-6s %-15s %s%s",
                j["jobId"],
                j["queueName"],
                j["status"],
                tries,
                j["scheduledDate"],
                events,
                name,
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ECaccess SOAP client for ECMWF Slurm job management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Environment variables:
              ECCERT              Path to .eccert.crt  (default: ~/.eccert.crt)
              https_ecaccess      HTTPS gateway host   (default: boaccess.ecmwf.int)
              http_ecaccess       HTTP gateway host    (default: boaccess.ecmwf.int)
        """),
    )
    parser.add_argument("--debug", action="store_true", help="Print raw SOAP envelopes")
    parser.add_argument("--cert", type=Path, default=None, help="Path to .eccert.crt")

    sub = parser.add_subparsers(dest="command", required=True)

    # gateway (no auth)
    sub.add_parser("gateway", help="Print the gateway name (no auth needed)")

    # ops
    sub.add_parser("ops", help="List available operations (validates certificate)")

    # queues
    p_q = sub.add_parser("queues", help="List available batch queues")
    p_q.add_argument("queue_name", nargs="?", help="Queue name for detail view")

    # jobs
    p_j = sub.add_parser("jobs", help="List your ECaccess jobs")
    p_j.add_argument("job_id", nargs="?", help="Job ID for detail view")

    # submit
    p_s = sub.add_parser("submit", help="Submit a Slurm job script")
    p_s.add_argument("script", help="Path to Slurm job script (local, or remote with --distant)")
    p_s.add_argument("--queue", "-q", default=None, help="Queue name")
    p_s.add_argument("--name", "-n", default=None, help="Job name")
    p_s.add_argument(
        "--distant",
        "-d",
        action="store_true",
        help="Script is already at ECMWF (ECaccess path like home:scripts/job.sh)",
    )
    p_s.add_argument("--event-ids", default=None, help="Event IDs (semicolon-separated)")
    p_s.add_argument(
        "--no-directives", action="store_true", help="Script has no scheduler directives"
    )
    p_s.add_argument("--stderr-to-stdout", action="store_true", help="Merge stderr into stdout")

    # delete
    p_d = sub.add_parser("delete", help="Delete/cancel an ECaccess job")
    p_d.add_argument("job_id", help="ECaccess job ID")

    # restart
    p_r = sub.add_parser("restart", help="Restart an ECaccess job")
    p_r.add_argument("job_id", help="ECaccess job ID")

    # get
    p_g = sub.add_parser("get", help="Get job output/error/input")
    p_g.add_argument("job_id", help="ECaccess job ID")
    p_g.add_argument("--error", action="store_true", help="Get error output")
    p_g.add_argument("--input", action="store_true", help="Get input script")
    p_g.add_argument("-o", "--output", type=Path, help="Write to file instead of stdout")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(message)s")

    # Resolve cert path: --cert flag > ECCERT env > default
    cert_path = args.cert or Path(os.environ.get("ECCERT", str(DEFAULT_CERT)))

    # Resolve gateway hosts from env (matching ECaccess.pm lines 105-114)
    https_host = os.environ.get("https_ecaccess", DEFAULT_GATEWAY)
    http_host = os.environ.get("http_ecaccess", DEFAULT_GATEWAY)

    client = ECaccessClient(
        cert_path=cert_path,
        https_host=https_host,
        http_host=http_host,
        debug=args.debug,
    )

    try:
        match args.command:
            case "gateway":
                logger.info("%s", client.get_gateway_name() or "(no response)")

            case "ops":
                _print_ops(client)

            case "queues":
                _print_queues(client, args.queue_name)

            case "jobs":
                _print_jobs(client, args.job_id)

            case "submit":
                job_id = client.submit_job(
                    args.script,
                    queue=args.queue,
                    job_name=args.name,
                    distant=args.distant,
                    event_ids=args.event_ids,
                    no_directives=args.no_directives,
                    stderr_to_stdout=args.stderr_to_stdout,
                )
                logger.info("%s", job_id)

            case "delete":
                client.delete_job(args.job_id)
                logger.info("Deleted %s", args.job_id)

            case "restart":
                client.restart_job(args.job_id)
                logger.info("Restarted %s", args.job_id)

            case "get":
                which = "error" if args.error else ("input" if args.input else "output")
                data = client.get_job_output(args.job_id, which=which)
                if args.output:
                    args.output.write_bytes(data)
                    logger.info("Written to %s", args.output)
                else:
                    sys.stdout.buffer.write(data)

    except FileNotFoundError as exc:
        logger.error("Error: %s", exc)
        return 1
    except RuntimeError as exc:
        logger.error("Error: %s", exc)
        return 1
    except httpx.ConnectError as exc:
        logger.error("Connection error: %s", exc)
        return 1
    finally:
        client.release_token()

    return 0
