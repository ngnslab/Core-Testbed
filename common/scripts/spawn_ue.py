import os
from utils import load_template, write_config
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--imsi", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--opc", required=True)
    parser.add_argument("--gnb-name", required=True)
    args = parser.parse_args()

    template = load_template("ue-template.yaml")

    content = template.safe_substitute(
        IMSI=args.imsi,
        KEY=args.key,
        OPC=args.opc,
        GNB_NAME=args.gnb_name,
    )

    write_config(args.output, content)
    print(f"[+] UE config created at {args.output}")

if __name__ == "__main__":
    main()
