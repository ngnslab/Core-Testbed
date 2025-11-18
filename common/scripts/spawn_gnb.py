import os
from utils import load_template, write_config

# 생성 예:
# python spawn_gnb.py --output ../ran-node/configs/gnb.yaml --id 101 --name RAN1-GNB
# python spawn_gnb.py --output ../core-node/configs/gnb.yaml --id 300 --name CORE-GNB --amf-ip 172.16.0.10

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--id", required=True, help="gNB ID")
    parser.add_argument("--name", required=True, help="gNB Name")
    parser.add_argument("--nci", default="0x000001")
    parser.add_argument("--amf-ip", default="")
    args = parser.parse_args()

    template = load_template("gnb-template.yaml")

    content = template.safe_substitute(
        NCI=args.nci,
        GNB_ID=args.id,
        GNB_NAME=args.name,
        AMF_IP=args.amf_ip,
    )

    write_config(args.output, content)
    print(f"[+] gNB config created at {args.output}")

if __name__ == "__main__":
    main()
