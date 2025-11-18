# UERANSIM 설치

### UERANSIM 공식 GIthub Container
```
docker pull ghcr.io/aligungr/ueransim:latest
```

### Dockerfile.ueransim으로 설치
```
docker build --platform=linux/amd64 -t ueransim-base -f Dockerfile.ueransim .
```

### 빌드 검증
아키텍쳐 체크
```
docker inspect ueransim-base --format='{{.Architecture}}'
```

