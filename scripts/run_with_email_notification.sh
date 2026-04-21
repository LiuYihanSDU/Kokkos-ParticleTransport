#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  scripts/run_with_email_notification.sh [options] -- command [args...]

Runs a command, writes stdout/stderr to a log file, sends an email notification when
the command starts, sends periodic status emails while the command is still running,
and sends a final email after the command exits. The notification path does not depend
on GPT or external Python packages.

Options:
  --log PATH           Log path. Defaults to benchmark_runs/email_notifications/TIMESTAMP_COMMAND.log.
  --to EMAILS          Recipient list. Comma or space separated. Env: KPT_NOTIFY_TO.
  --from EMAIL         Sender address. Env: KPT_NOTIFY_FROM.
  --subject TEXT       Subject prefix. The final status is appended automatically.
  --tail-lines N       Number of log tail lines included in the email body. Default: 160.
  --status-interval-hours N
                       Periodic status interval in hours. Default: 12. Use 0 to disable.
  --no-attach          Do not attach the log file.
  --dry-run-email      Build and print the email payload, but do not send it.
  -h, --help           Show this help.

SMTP environment:
  KPT_NOTIFY_SMTP_HOST             SMTP server host.
  KPT_NOTIFY_SMTP_CONNECT_HOST     Optional TCP connect host/IP when DNS is proxied.
  KPT_NOTIFY_SMTP_PORT             SMTP server port. Defaults to 465 for SSL, 587 otherwise.
  KPT_NOTIFY_SMTP_USER             SMTP user name.
  KPT_NOTIFY_SMTP_PASSWORD         SMTP password or app authorization code.
  KPT_NOTIFY_SMTP_PASSWORD_FILE    File containing the SMTP password.
  KPT_NOTIFY_SMTP_SSL              Use implicit SSL. Default: 0.
  KPT_NOTIFY_SMTP_STARTTLS         Use STARTTLS when SSL is disabled. Default: 1.

Other environment:
  KPT_NOTIFY_ATTACH_MAX_MB         Maximum attachment size. Default: 15.
  KPT_NOTIFY_GZIP_LARGE_LOG        Try gzip attachment for oversized logs. Default: 1.
  KPT_NOTIFY_GZIP_INPUT_MAX_MB     Largest raw log considered for gzip. Default: 200.
  KPT_NOTIFY_FAILS_JOB             Make email failure change wrapper exit code. Default: 0.
  KPT_NOTIFY_STATUS_INTERVAL_HOURS Periodic status interval in hours. Default: 12.
  KPT_NOTIFY_STATUS_INTERVAL_SECONDS
                                   Test-only override for the periodic interval.
  KPT_NOTIFY_SMTP_TIMEOUT_SECONDS  SMTP connection timeout. Default: 60.

Example:
  KPT_NOTIFY_FROM=202421417@mail.sdu.edu.cn \
  KPT_NOTIFY_SMTP_HOST=smtp.qiye.163.com \
  KPT_NOTIFY_SMTP_CONNECT_HOST=139.95.4.241 \
  KPT_NOTIFY_SMTP_PORT=465 \
  KPT_NOTIFY_SMTP_SSL=1 \
  KPT_NOTIFY_SMTP_STARTTLS=0 \
  KPT_NOTIFY_SMTP_USER=202421417@mail.sdu.edu.cn \
  KPT_NOTIFY_SMTP_PASSWORD_FILE=$HOME/.config/kpt_smtp_password \
  scripts/run_with_email_notification.sh \
      --subject "Parker movie run" \
      --status-interval-hours 12 \
      --log benchmark_runs/parker_movie/run_with_notify.log \
      -- scripts/run_kokkos_cpu_parker_movie_pipeline.sh
EOF
}

absolute_path() {
    local path="$1"
    if [[ "${path}" = /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s\n' "${REPO_ROOT}/${path}"
    fi
}

format_command() {
    local arg
    local quoted
    local command_text=""
    for arg in "$@"; do
        printf -v quoted '%q' "${arg}"
        command_text+="${quoted} "
    done
    printf '%s\n' "${command_text% }"
}

sanitize_name() {
    local name="$1"
    name="${name##*/}"
    name="${name//[^A-Za-z0-9_.-]/_}"
    if [[ -z "${name}" ]]; then
        name="command"
    fi
    printf '%s\n' "${name}"
}

log_path="${KPT_NOTIFY_LOG_PATH:-}"
recipient_list="${KPT_NOTIFY_TO:-liu-yh@outlook.com}"
sender_address="${KPT_NOTIFY_FROM:-}"
subject_prefix="${KPT_NOTIFY_SUBJECT:-Kokkos particle transport job}"
tail_lines="${KPT_NOTIFY_TAIL_LINES:-160}"
attach_log="${KPT_NOTIFY_ATTACH_LOG:-1}"
dry_run_email="${KPT_NOTIFY_DRY_RUN:-0}"
status_interval_hours="${KPT_NOTIFY_STATUS_INTERVAL_HOURS:-12}"
status_interval_seconds="${KPT_NOTIFY_STATUS_INTERVAL_SECONDS:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --log)
            [[ $# -ge 2 ]] || { echo "--log requires a path" >&2; exit 2; }
            log_path="$2"
            shift 2
            ;;
        --to)
            [[ $# -ge 2 ]] || { echo "--to requires an address list" >&2; exit 2; }
            recipient_list="$2"
            shift 2
            ;;
        --from)
            [[ $# -ge 2 ]] || { echo "--from requires an address" >&2; exit 2; }
            sender_address="$2"
            shift 2
            ;;
        --subject)
            [[ $# -ge 2 ]] || { echo "--subject requires text" >&2; exit 2; }
            subject_prefix="$2"
            shift 2
            ;;
        --tail-lines)
            [[ $# -ge 2 ]] || { echo "--tail-lines requires a number" >&2; exit 2; }
            tail_lines="$2"
            shift 2
            ;;
        --status-interval-hours)
            [[ $# -ge 2 ]] || { echo "--status-interval-hours requires a number" >&2; exit 2; }
            status_interval_hours="$2"
            status_interval_seconds=""
            shift 2
            ;;
        --no-attach)
            attach_log="0"
            shift
            ;;
        --dry-run-email)
            dry_run_email="1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi

if ! [[ "${tail_lines}" =~ ^[0-9]+$ ]]; then
    echo "--tail-lines must be a non-negative integer." >&2
    exit 2
fi

if [[ -n "${status_interval_seconds}" ]]; then
    if ! [[ "${status_interval_seconds}" =~ ^[0-9]+$ ]]; then
        echo "KPT_NOTIFY_STATUS_INTERVAL_SECONDS must be a non-negative integer." >&2
        exit 2
    fi
else
    if ! [[ "${status_interval_hours}" =~ ^[0-9]+$ ]]; then
        echo "--status-interval-hours must be a non-negative integer." >&2
        exit 2
    fi
    status_interval_seconds=$((status_interval_hours * 3600))
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
command_name="$(sanitize_name "$1")"
if [[ -z "${log_path}" ]]; then
    log_path="${REPO_ROOT}/benchmark_runs/email_notifications/${timestamp}_${command_name}.log"
else
    log_path="$(absolute_path "${log_path}")"
fi

mkdir -p "$(dirname "${log_path}")"
: > "${log_path}"

command_text="$(format_command "$@")"
host_name="$(hostname -f 2>/dev/null || hostname)"
run_cwd="$(pwd -P)"
start_iso="$(date -Is)"
start_epoch="$(date +%s)"

{
    printf 'Command: %s\n' "${command_text}"
    printf 'Host: %s\n' "${host_name}"
    printf 'Working directory: %s\n' "${run_cwd}"
    printf 'Repository: %s\n' "${REPO_ROOT}"
    printf 'Start time: %s\n' "${start_iso}"
    if [[ "${status_interval_seconds}" -gt 0 ]]; then
        printf 'Status interval seconds: %s\n' "${status_interval_seconds}"
    else
        printf 'Status interval seconds: disabled\n'
    fi
    printf '%s\n' '--- command output begins ---'
} | tee -a "${log_path}"

send_notification() {
    local phase="$1"
    local phase_exit_code="${2:-}"
    local phase_end_iso="${3:-}"
    local next_status_iso="${4:-}"
    local now_epoch
    local duration_seconds
    local log_bytes
    local notification_status

    now_epoch="$(date +%s)"
    duration_seconds=$((now_epoch - start_epoch))
    log_bytes="$(wc -c < "${log_path}" | tr -d '[:space:]')"

    notification_status=0
    KPT_NOTIFY_TO="${recipient_list}" \
    KPT_NOTIFY_FROM="${sender_address}" \
    KPT_NOTIFY_SUBJECT="${subject_prefix}" \
    KPT_NOTIFY_COMMAND="${command_text}" \
    KPT_NOTIFY_HOST="${host_name}" \
    KPT_NOTIFY_RUN_CWD="${run_cwd}" \
    KPT_NOTIFY_REPO_ROOT="${REPO_ROOT}" \
    KPT_NOTIFY_LOG_PATH="${log_path}" \
    KPT_NOTIFY_PHASE="${phase}" \
    KPT_NOTIFY_COMMAND_PID="${command_pid:-}" \
    KPT_NOTIFY_EXIT_CODE="${phase_exit_code}" \
    KPT_NOTIFY_START_ISO="${start_iso}" \
    KPT_NOTIFY_END_ISO="${phase_end_iso}" \
    KPT_NOTIFY_DURATION_SECONDS="${duration_seconds}" \
    KPT_NOTIFY_LOG_BYTES="${log_bytes}" \
    KPT_NOTIFY_NEXT_STATUS_ISO="${next_status_iso}" \
    KPT_NOTIFY_STATUS_INTERVAL_SECONDS="${status_interval_seconds}" \
    KPT_NOTIFY_TAIL_LINES="${tail_lines}" \
    KPT_NOTIFY_ATTACH_LOG="${attach_log}" \
    KPT_NOTIFY_DRY_RUN="${dry_run_email}" \
    python3 <<'PY' || notification_status=$?
from __future__ import annotations

import gzip
import os
import shutil
import smtplib
import socket
import ssl
import subprocess
import sys
from collections import deque
from email.message import EmailMessage
from pathlib import Path


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def split_addresses(value: str) -> list[str]:
    return [part.strip() for part in value.replace(",", " ").split() if part.strip()]


def read_password() -> str:
    password_file = os.environ.get("KPT_NOTIFY_SMTP_PASSWORD_FILE", "").strip()
    if password_file:
        return Path(password_file).expanduser().read_text(encoding="utf-8").strip()
    return os.environ.get("KPT_NOTIFY_SMTP_PASSWORD", "")


def tail_text(path: Path, line_count: int) -> str:
    if line_count <= 0 or not path.exists():
        return ""
    lines: deque[str] = deque(maxlen=line_count)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lines.append(line)
    return "".join(lines)


def add_log_attachment(message: EmailMessage, path: Path) -> str:
    if not env_bool("KPT_NOTIFY_ATTACH_LOG", True):
        return "Log attachment disabled."
    if not path.exists():
        return "Log attachment skipped because the log file does not exist."

    max_mb = float(os.environ.get("KPT_NOTIFY_ATTACH_MAX_MB", "15"))
    max_bytes = int(max_mb * 1024 * 1024)
    log_size = path.stat().st_size

    if log_size <= max_bytes:
        text = path.read_text(encoding="utf-8", errors="replace")
        message.add_attachment(text, subtype="plain", filename=path.name)
        return f"Attached log file ({log_size} bytes)."

    if env_bool("KPT_NOTIFY_GZIP_LARGE_LOG", True):
        input_max_mb = float(os.environ.get("KPT_NOTIFY_GZIP_INPUT_MAX_MB", "200"))
        input_max_bytes = int(input_max_mb * 1024 * 1024)
        if log_size <= input_max_bytes:
            compressed = gzip.compress(path.read_bytes())
            if len(compressed) <= max_bytes:
                message.add_attachment(
                    compressed,
                    maintype="application",
                    subtype="gzip",
                    filename=f"{path.name}.gz",
                )
                return (
                    f"Attached gzip-compressed log "
                    f"({log_size} raw bytes, {len(compressed)} compressed bytes)."
                )

    return (
        f"Log attachment skipped because the log is {log_size} bytes, "
        f"larger than the {max_bytes} byte attachment limit."
    )


def build_message() -> EmailMessage:
    to_addresses = split_addresses(os.environ.get("KPT_NOTIFY_TO", ""))
    dry_run = env_bool("KPT_NOTIFY_DRY_RUN", False)
    if not to_addresses and dry_run:
        to_addresses = ["dry-run@example.invalid"]
    if not to_addresses:
        raise RuntimeError("KPT_NOTIFY_TO or --to is required.")

    smtp_user = os.environ.get("KPT_NOTIFY_SMTP_USER", "").strip()
    sender = os.environ.get("KPT_NOTIFY_FROM", "").strip() or smtp_user
    if not sender:
        sender = f"{os.environ.get('USER', 'kpt')}@{socket.getfqdn()}"

    phase = os.environ.get("KPT_NOTIFY_PHASE", "final").strip().lower()
    exit_code_text = os.environ.get("KPT_NOTIFY_EXIT_CODE", "").strip()
    if phase == "start":
        status_label = "STARTED"
    elif phase == "status":
        status_label = "RUNNING"
    else:
        exit_code = int(exit_code_text or "0")
        status_label = "SUCCESS" if exit_code == 0 else f"FAILED exit={exit_code}"
    subject_prefix = os.environ.get("KPT_NOTIFY_SUBJECT", "Kokkos particle transport job")
    subject = f"{subject_prefix}: {status_label}"
    log_path = Path(os.environ["KPT_NOTIFY_LOG_PATH"])
    tail_lines = int(os.environ.get("KPT_NOTIFY_TAIL_LINES", "160"))
    log_tail = tail_text(log_path, tail_lines)

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = ", ".join(to_addresses)

    body = [
        f"Status: {status_label}",
        f"Phase: {phase}",
        f"Host: {os.environ.get('KPT_NOTIFY_HOST', '')}",
        f"Working directory: {os.environ.get('KPT_NOTIFY_RUN_CWD', '')}",
        f"Repository: {os.environ.get('KPT_NOTIFY_REPO_ROOT', '')}",
        f"Command: {os.environ.get('KPT_NOTIFY_COMMAND', '')}",
        f"Command PID: {os.environ.get('KPT_NOTIFY_COMMAND_PID', '')}",
        f"Start: {os.environ.get('KPT_NOTIFY_START_ISO', '')}",
        f"End: {os.environ.get('KPT_NOTIFY_END_ISO', '') or 'not finished'}",
        f"Duration seconds: {os.environ.get('KPT_NOTIFY_DURATION_SECONDS', '')}",
        f"Next automatic status: {os.environ.get('KPT_NOTIFY_NEXT_STATUS_ISO', '') or 'none'}",
        f"Log path: {log_path}",
        f"Log bytes: {os.environ.get('KPT_NOTIFY_LOG_BYTES', '')}",
        "",
        f"Last {tail_lines} log lines:",
        "```",
        log_tail,
        "```",
    ]
    message.set_content("\n".join(body))
    attachment_status = add_log_attachment(message, log_path)
    message["X-KPT-Attachment-Status"] = attachment_status
    return message


def send_message(message: EmailMessage) -> None:
    if env_bool("KPT_NOTIFY_DRY_RUN", False):
        print("Dry-run email payload follows.", file=sys.stderr)
        print(message.as_string())
        return

    sendmail_path = shutil.which("sendmail")
    smtp_host = os.environ.get("KPT_NOTIFY_SMTP_HOST", "").strip()
    smtp_connect_host = os.environ.get("KPT_NOTIFY_SMTP_CONNECT_HOST", "").strip() or smtp_host
    if not smtp_host and sendmail_path:
        subprocess.run([sendmail_path, "-t", "-oi"], input=message.as_bytes(), check=True)
        return
    if not smtp_host:
        raise RuntimeError(
            "No local sendmail command was found, and KPT_NOTIFY_SMTP_HOST is not set."
        )

    use_ssl = env_bool("KPT_NOTIFY_SMTP_SSL", False)
    default_port = "465" if use_ssl else "587"
    smtp_port = int(os.environ.get("KPT_NOTIFY_SMTP_PORT", default_port))
    smtp_user = os.environ.get("KPT_NOTIFY_SMTP_USER", "").strip()
    smtp_password = read_password()
    smtp_timeout = float(os.environ.get("KPT_NOTIFY_SMTP_TIMEOUT_SECONDS", "60"))

    if use_ssl:
        context = ssl.create_default_context()
        if smtp_connect_host != smtp_host:
            class SMTPSSLConnectHost(smtplib.SMTP_SSL):
                def _get_socket(self, host, port, timeout):
                    raw_socket = socket.create_connection(
                        (smtp_connect_host, port),
                        smtp_timeout,
                        self.source_address,
                    )
                    return self.context.wrap_socket(raw_socket, server_hostname=smtp_host)

            smtp_class = SMTPSSLConnectHost
        else:
            smtp_class = smtplib.SMTP_SSL
        with smtp_class(smtp_host, smtp_port, timeout=smtp_timeout, context=context) as server:
            if smtp_user:
                server.login(smtp_user, smtp_password)
            server.send_message(message)
    else:
        if smtp_connect_host != smtp_host:
            server = smtplib.SMTP(timeout=smtp_timeout)
            server._host = smtp_host
            server.connect(smtp_connect_host, smtp_port)
        else:
            server = smtplib.SMTP(smtp_host, smtp_port, timeout=smtp_timeout)
        with server:
            server.ehlo()
            if env_bool("KPT_NOTIFY_SMTP_STARTTLS", True):
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
            if smtp_user:
                server.login(smtp_user, smtp_password)
            server.send_message(message)


def main() -> int:
    message = build_message()
    send_message(message)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Email notification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
PY

    if [[ "${notification_status}" -ne 0 ]]; then
        error_path="${log_path}.notification_error"
        {
            printf 'Email notification phase %s failed with exit code %s.\n' "${phase}" "${notification_status}"
            printf 'Check SMTP settings or use --dry-run-email to inspect the payload.\n'
        } | tee -a "${error_path}" >&2
    fi
    return "${notification_status}"
}

next_status_iso=""
if [[ "${status_interval_seconds}" -gt 0 ]]; then
    next_status_iso="$(date -Is -d "@$((start_epoch + status_interval_seconds))")"
fi

command_status_file="$(mktemp "${TMPDIR:-/tmp}/kpt_notify_status.XXXXXX")"
command_done_file="$(mktemp "${TMPDIR:-/tmp}/kpt_notify_done.XXXXXX")"
rm -f "${command_status_file}" "${command_done_file}"
command_pid=""
watchdog_pid=""

cleanup_state() {
    if [[ -n "${watchdog_pid}" ]]; then
        kill "${watchdog_pid}" 2>/dev/null || true
    fi
    rm -f "${command_status_file}" "${command_done_file}"
}
trap cleanup_state EXIT
trap 'if [[ -n "${command_pid}" ]]; then kill "${command_pid}" 2>/dev/null || true; fi; exit 130' INT TERM

(
    set +e
    "$@" 2>&1 | tee -a "${log_path}"
    pipeline_status=${PIPESTATUS[0]}
    printf '%s\n' "${pipeline_status}" > "${command_status_file}"
    touch "${command_done_file}"
    exit "${pipeline_status}"
) &
command_pid=$!

send_notification "start" "" "" "${next_status_iso}" || true

if [[ "${status_interval_seconds}" -gt 0 ]]; then
    (
        while true; do
            sleep "${status_interval_seconds}"
            if [[ -e "${command_done_file}" ]]; then
                exit 0
            fi
            next_status_iso="$(date -Is -d "@$(($(date +%s) + status_interval_seconds))")"
            send_notification "status" "" "" "${next_status_iso}" || true
        done
    ) &
    watchdog_pid=$!
fi

set +e
wait "${command_pid}"
command_status=$?
set -e

if [[ -s "${command_status_file}" ]]; then
    command_status="$(tr -d '[:space:]' < "${command_status_file}")"
fi

if [[ -n "${watchdog_pid}" ]]; then
    kill "${watchdog_pid}" 2>/dev/null || true
    wait "${watchdog_pid}" 2>/dev/null || true
    watchdog_pid=""
fi

end_iso="$(date -Is)"
end_epoch="$(date +%s)"
duration_seconds=$((end_epoch - start_epoch))

{
    printf '%s\n' '--- command output ends ---'
    printf 'End time: %s\n' "${end_iso}"
    printf 'Duration seconds: %s\n' "${duration_seconds}"
    printf 'Exit code: %s\n' "${command_status}"
    printf 'Log path: %s\n' "${log_path}"
} | tee -a "${log_path}"

final_notification_status=0
send_notification "final" "${command_status}" "${end_iso}" "" || final_notification_status=$?
if [[ "${final_notification_status}" -ne 0 && "${KPT_NOTIFY_FAILS_JOB:-0}" == "1" && "${command_status}" -eq 0 ]]; then
    exit "${final_notification_status}"
fi

exit "${command_status}"
