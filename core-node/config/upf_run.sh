#!/bin/bash

# 1. ogstun 장치(TUN 인터페이스) 미리 생성
if ! ip link show ogstun > /dev/null 2>&1; then
    ip tuntap add name ogstun mode tun
fi

# 2. IP 주소 할당 및 인터페이스 켜기 (사용자가 수동으로 했던 명령어)
ip addr add 10.45.0.1/16 dev ogstun
ip link set ogstun up

# (선택 사항) IPv6 주소도 필요하다면 추가
ip addr add 2001:db8:cafe::1/48 dev ogstun

# IP 포워딩 활성화 및 NAT 설정
sysctl -w net.ipv4.ip_forward=1 # 이건 필요없는 것 같기도..
iptables -t nat -A POSTROUTING -s 10.45.0.0/16 ! -o ogstun -j MASQUERADE # 이거하면 됨

# 3. 설정 완료 후 UPF 실행
echo "✅ Network setup done. Starting UPF..."
exec open5gs-upfd -c /open5gs/config/upf.yaml