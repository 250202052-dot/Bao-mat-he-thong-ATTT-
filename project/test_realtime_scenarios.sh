#!/usr/bin/env bash
set -euo pipefail

TARGET_IP="${TARGET_IP:-192.168.124.132}"
WEB_URL="${WEB_URL:-http://$TARGET_IP}"
SSH_TARGET="${SSH_TARGET:-}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-5}"
UDP_TOP_PORTS="${UDP_TOP_PORTS:-200}"
TCP_TOP_PORTS="${TCP_TOP_PORTS:-1000}"
HPING_PORT="${HPING_PORT:-80}"
HPING_INTERVAL_US="${HPING_INTERVAL_US:-10000}"
HPING_COUNT="${HPING_COUNT:-200}"
PCAP_IFACE="${PCAP_IFACE:-eth0}"
PCAP_FILE="${PCAP_FILE:-}"
PCAP_DIR="${PCAP_DIR:-}"
PCAP_GLOB="${PCAP_GLOB:-*.pcap}"
TCPREPLAY_MBPS="${TCPREPLAY_MBPS:-10}"

usage() {
  cat <<'EOF'
Usage:
  ./test_realtime_scenarios.sh <scenario>

Scenarios:
  benign_ping       ICMP ping only
  benign_web        Simple HTTP request
  benign_ssh        SSH connection attempt (requires SSH_TARGET=user@host or target with user preset)
  recon_tcp         TCP SYN scan (top ports)
  recon_service     TCP service/version detection
  recon_full        Full TCP aggressive scan
  udp_scan          UDP top-ports scan
  dos_light         Light SYN flood in lab only
  replay_pcap       Replay one PCAP file (requires PCAP_FILE)
  replay_attack_set Replay all PCAPs matching PCAP_GLOB in PCAP_DIR
  mixed             Benign + TCP recon + UDP recon
  all               Run a recommended sequence of scenarios

Environment overrides:
  TARGET_IP=192.168.124.132
  WEB_URL=http://192.168.124.132
  SSH_TARGET=user@192.168.124.132
  SLEEP_BETWEEN=5
  TCP_TOP_PORTS=1000
  UDP_TOP_PORTS=200
  HPING_PORT=80
  HPING_INTERVAL_US=10000
  HPING_COUNT=200
  PCAP_IFACE=eth0
  PCAP_FILE=/path/to/sample_attack.pcap
  PCAP_DIR=/path/to/pcaps
  PCAP_GLOB=*.pcap
  TCPREPLAY_MBPS=10

Examples:
  TARGET_IP=192.168.124.132 ./test_realtime_scenarios.sh recon_tcp
  TARGET_IP=192.168.124.132 UDP_TOP_PORTS=100 ./test_realtime_scenarios.sh udp_scan
  PCAP_FILE=/data/attacks/exploit_1.pcap ./test_realtime_scenarios.sh replay_pcap
  PCAP_DIR=/data/attacks PCAP_GLOB='*.pcapng' ./test_realtime_scenarios.sh replay_attack_set
  TARGET_IP=192.168.124.132 SSH_TARGET=user@192.168.124.132 ./test_realtime_scenarios.sh all
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

pause_between() {
  sleep "$SLEEP_BETWEEN"
}

run_benign_ping() {
  need_cmd ping
  log "Running benign ICMP ping to $TARGET_IP"
  ping -c 10 "$TARGET_IP"
}

run_benign_web() {
  need_cmd curl
  log "Running benign HTTP request to $WEB_URL"
  curl --max-time 10 --silent --show-error "$WEB_URL" >/dev/null || true
}

run_benign_ssh() {
  need_cmd ssh
  if [[ -z "$SSH_TARGET" ]]; then
    echo "SSH_TARGET is required for benign_ssh, e.g. SSH_TARGET=user@$TARGET_IP" >&2
    exit 1
  fi
  log "Running benign SSH connectivity test to $SSH_TARGET"
  ssh -o ConnectTimeout=5 -o BatchMode=yes "$SSH_TARGET" exit || true
}

run_recon_tcp() {
  need_cmd nmap
  log "Running TCP SYN reconnaissance scan against $TARGET_IP"
  nmap -sS --top-ports "$TCP_TOP_PORTS" "$TARGET_IP"
}

run_recon_service() {
  need_cmd nmap
  log "Running TCP service/version scan against $TARGET_IP"
  nmap -sS -sV --top-ports 100 "$TARGET_IP"
}

run_recon_full() {
  need_cmd nmap
  log "Running full TCP aggressive scan against $TARGET_IP"
  nmap -p- -A "$TARGET_IP"
}

run_udp_scan() {
  need_cmd nmap
  log "Running UDP scan against $TARGET_IP (top ports: $UDP_TOP_PORTS)"
  nmap -sU --top-ports "$UDP_TOP_PORTS" "$TARGET_IP"
}

run_dos_light() {
  need_cmd sudo
  need_cmd hping3
  log "Running light SYN flood in lab against $TARGET_IP:$HPING_PORT"
  sudo hping3 -S -p "$HPING_PORT" -i "u$HPING_INTERVAL_US" -c "$HPING_COUNT" "$TARGET_IP"
}

run_replay_pcap() {
  need_cmd sudo
  need_cmd tcpreplay
  if [[ -z "$PCAP_FILE" ]]; then
    echo "PCAP_FILE is required for replay_pcap" >&2
    exit 1
  fi
  if [[ ! -f "$PCAP_FILE" ]]; then
    echo "PCAP_FILE does not exist: $PCAP_FILE" >&2
    exit 1
  fi
  log "Replaying PCAP on $PCAP_IFACE at ${TCPREPLAY_MBPS}Mbps: $PCAP_FILE"
  sudo tcpreplay --intf1="$PCAP_IFACE" --mbps="$TCPREPLAY_MBPS" "$PCAP_FILE"
}

run_replay_attack_set() {
  if [[ -z "$PCAP_DIR" ]]; then
    echo "PCAP_DIR is required for replay_attack_set" >&2
    exit 1
  fi
  if [[ ! -d "$PCAP_DIR" ]]; then
    echo "PCAP_DIR does not exist: $PCAP_DIR" >&2
    exit 1
  fi

  shopt -s nullglob
  local files=("$PCAP_DIR"/$PCAP_GLOB)
  shopt -u nullglob

  if [[ ${#files[@]} -eq 0 ]]; then
    echo "No PCAP files found in $PCAP_DIR matching $PCAP_GLOB" >&2
    exit 1
  fi

  local file
  for file in "${files[@]}"; do
    log "Replaying attack PCAP: $file"
    PCAP_FILE="$file" run_replay_pcap
    pause_between
  done
}

run_mixed() {
  run_benign_ping
  pause_between
  run_recon_tcp
  pause_between
  run_udp_scan
  pause_between
  run_benign_web
}

run_all() {
  log "Starting full recommended realtime IDS test sequence"
  run_benign_ping
  pause_between
  run_benign_web
  pause_between
  if [[ -n "$SSH_TARGET" ]]; then
    run_benign_ssh
    pause_between
  fi
  run_recon_tcp
  pause_between
  run_recon_service
  pause_between
  run_udp_scan
  pause_between
  run_dos_light
  if [[ -n "$PCAP_DIR" || -n "$PCAP_FILE" ]]; then
    pause_between
    if [[ -n "$PCAP_FILE" ]]; then
      run_replay_pcap
    else
      run_replay_attack_set
    fi
  fi
  log "Finished full recommended realtime IDS test sequence"
}

main() {
  local scenario="${1:-}"
  if [[ -z "$scenario" || "$scenario" == "-h" || "$scenario" == "--help" ]]; then
    usage
    exit 0
  fi

  case "$scenario" in
    benign_ping) run_benign_ping ;;
    benign_web) run_benign_web ;;
    benign_ssh) run_benign_ssh ;;
    recon_tcp) run_recon_tcp ;;
    recon_service) run_recon_service ;;
    recon_full) run_recon_full ;;
    udp_scan) run_udp_scan ;;
    dos_light) run_dos_light ;;
    replay_pcap) run_replay_pcap ;;
    replay_attack_set) run_replay_attack_set ;;
    mixed) run_mixed ;;
    all) run_all ;;
    *)
      echo "Unknown scenario: $scenario" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
