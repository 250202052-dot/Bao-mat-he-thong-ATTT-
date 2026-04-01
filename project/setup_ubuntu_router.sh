#!/usr/bin/env bash
set -euo pipefail

# Ubuntu router bootstrap for:
# - 1 WAN NIC on NAT (DHCP)
# - 1 LAN NIC on vmnet1: 10.0.0.0/8
# - 1 LAN NIC on vmnet2: 172.16.0.0/16
#
# Default interface names are examples only. Override them when running:
#   sudo WAN_IF=ens33 LAN1_IF=ens34 LAN2_IF=ens35 ./setup_ubuntu_router.sh
#
# Optional DHCP via dnsmasq:
#   sudo INSTALL_DNSMASQ=1 WAN_IF=ens33 LAN1_IF=ens34 LAN2_IF=ens35 ./setup_ubuntu_router.sh

WAN_IF="${WAN_IF:-ens33}"
LAN1_IF="${LAN1_IF:-ens34}"
LAN2_IF="${LAN2_IF:-ens35}"

LAN1_ADDR="${LAN1_ADDR:-10.0.0.1/8}"
LAN2_ADDR="${LAN2_ADDR:-172.16.0.1/16}"

LAN1_NET_CIDR="${LAN1_NET_CIDR:-10.0.0.0/8}"
LAN2_NET_CIDR="${LAN2_NET_CIDR:-172.16.0.0/16}"

LAN1_DHCP_RANGE="${LAN1_DHCP_RANGE:-10.0.0.100,10.0.0.200,255.0.0.0,12h}"
LAN2_DHCP_RANGE="${LAN2_DHCP_RANGE:-172.16.0.100,172.16.0.200,255.255.0.0,12h}"

NETPLAN_FILE="${NETPLAN_FILE:-/etc/netplan/99-router.yaml}"
SYSCTL_FILE="${SYSCTL_FILE:-/etc/sysctl.d/99-router.conf}"
IPTABLES_RULES_V4="${IPTABLES_RULES_V4:-/etc/iptables/rules.v4}"
DNSMASQ_FILE="${DNSMASQ_FILE:-/etc/dnsmasq.d/router.conf}"

INSTALL_DNSMASQ="${INSTALL_DNSMASQ:-0}"
INSTALL_IPTABLES_PERSISTENT="${INSTALL_IPTABLES_PERSISTENT:-1}"

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "Please run as root: sudo $0" >&2
    exit 1
  fi
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

backup_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    cp "$file" "${file}.bak.$(date +%Y%m%d_%H%M%S)"
  fi
}

validate_iface() {
  local iface="$1"
  if ! ip link show "$iface" >/dev/null 2>&1; then
    echo "Interface not found: $iface" >&2
    ip -br link || true
    exit 1
  fi
}

write_netplan() {
  backup_file "$NETPLAN_FILE"
  cat > "$NETPLAN_FILE" <<EOF
network:
  version: 2
  renderer: networkd
  ethernets:
    ${WAN_IF}:
      dhcp4: true
    ${LAN1_IF}:
      dhcp4: false
      addresses:
        - ${LAN1_ADDR}
    ${LAN2_IF}:
      dhcp4: false
      addresses:
        - ${LAN2_ADDR}
EOF
}

write_sysctl() {
  backup_file "$SYSCTL_FILE"
  cat > "$SYSCTL_FILE" <<'EOF'
net.ipv4.ip_forward=1
net.ipv4.conf.all.rp_filter=0
net.ipv4.conf.default.rp_filter=0
EOF
  sysctl --system >/dev/null
}

install_packages() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  if [[ "$INSTALL_IPTABLES_PERSISTENT" == "1" ]]; then
    apt-get install -y iptables-persistent
  fi
  if [[ "$INSTALL_DNSMASQ" == "1" ]]; then
    apt-get install -y dnsmasq
  fi
}

apply_netplan() {
  netplan generate
  netplan apply
  sleep 2
}

apply_iptables() {
  iptables -F
  iptables -t nat -F
  iptables -X

  iptables -P INPUT ACCEPT
  iptables -P FORWARD DROP
  iptables -P OUTPUT ACCEPT

  iptables -A FORWARD -i "$WAN_IF" -o "$LAN1_IF" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
  iptables -A FORWARD -i "$WAN_IF" -o "$LAN2_IF" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
  iptables -A FORWARD -i "$LAN1_IF" -o "$WAN_IF" -j ACCEPT
  iptables -A FORWARD -i "$LAN2_IF" -o "$WAN_IF" -j ACCEPT

  # Allow routing between the two internal labs as well.
  iptables -A FORWARD -i "$LAN1_IF" -o "$LAN2_IF" -j ACCEPT
  iptables -A FORWARD -i "$LAN2_IF" -o "$LAN1_IF" -j ACCEPT

  iptables -t nat -A POSTROUTING -s "$LAN1_NET_CIDR" -o "$WAN_IF" -j MASQUERADE
  iptables -t nat -A POSTROUTING -s "$LAN2_NET_CIDR" -o "$WAN_IF" -j MASQUERADE

  mkdir -p "$(dirname "$IPTABLES_RULES_V4")"
  iptables-save > "$IPTABLES_RULES_V4"
}

write_dnsmasq() {
  backup_file "$DNSMASQ_FILE"
  cat > "$DNSMASQ_FILE" <<EOF
bind-interfaces
interface=${LAN1_IF}
interface=${LAN2_IF}

dhcp-option=option:router,10.0.0.1
dhcp-range=${LAN1_DHCP_RANGE}

dhcp-option=tag:${LAN2_IF},option:router,172.16.0.1
dhcp-range=${LAN2_IF},${LAN2_DHCP_RANGE}
EOF

  systemctl restart dnsmasq
  systemctl enable dnsmasq
}

print_summary() {
  cat <<EOF

Router configuration complete.

Interfaces:
  WAN  (${WAN_IF}) -> DHCP via NAT
  LAN1 (${LAN1_IF}) -> ${LAN1_ADDR}  [vmnet1]
  LAN2 (${LAN2_IF}) -> ${LAN2_ADDR}  [vmnet2]

Internal networks:
  vmnet1 -> ${LAN1_NET_CIDR}
  vmnet2 -> ${LAN2_NET_CIDR}

Routing:
  - IPv4 forwarding enabled
  - NAT enabled from LAN1/LAN2 out through ${WAN_IF}
  - Inter-LAN forwarding enabled between ${LAN1_IF} and ${LAN2_IF}

Next steps on client VMs:
  vmnet1 host IP example: 10.0.0.10/8, gateway 10.0.0.1
  vmnet2 host IP example: 172.16.0.10/16, gateway 172.16.0.1

Useful checks:
  ip -br addr
  ip route
  sysctl net.ipv4.ip_forward
  iptables -S
  iptables -t nat -S
EOF
}

main() {
  require_root
  need_cmd ip
  need_cmd netplan
  need_cmd iptables

  validate_iface "$WAN_IF"
  validate_iface "$LAN1_IF"
  validate_iface "$LAN2_IF"

  log "Installing required packages"
  install_packages

  log "Writing netplan config to $NETPLAN_FILE"
  write_netplan

  log "Applying netplan"
  apply_netplan

  log "Enabling IPv4 forwarding"
  write_sysctl

  log "Applying iptables and NAT rules"
  apply_iptables

  if [[ "$INSTALL_DNSMASQ" == "1" ]]; then
    log "Configuring dnsmasq DHCP for internal networks"
    write_dnsmasq
  fi

  print_summary
}

main "$@"
