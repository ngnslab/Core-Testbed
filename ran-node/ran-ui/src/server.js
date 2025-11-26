import express from "express";
import morgan from "morgan";
import path from "path";
import { fileURLToPath } from "url";
import { MongoClient } from "mongodb";
import Docker from "dockerode";
import createUeRouter from "./routes/ue.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 7777;
const HOST = process.env.HOST || "0.0.0.0";

app.use(express.json());
app.use(morgan("dev"));

// MongoDB와 Docker 클라이언트를 미리 초기화해 라우터에 주입
const mongoUri = process.env.MONGO_URI || "mongodb://o5gs-mongo/open5gs";
const mongoClient = new MongoClient(mongoUri);
await mongoClient.connect();
const db = mongoClient.db();
const subscribers = db.collection("subscribers");

// 도커 관련 설정은 환경 변수로 치환 가능하도록 묶어둔다
const dockerSocket = process.env.DOCKER_SOCKET || "/var/run/docker.sock";
const docker = new Docker({ socketPath: dockerSocket });

const configMountDir = process.env.CONFIG_MOUNT_DIR || "/configs";
const configHostDir = process.env.CONFIG_HOST_DIR || configMountDir;
const ranNetwork = process.env.RAN_NETWORK || "ran-node_rannet";
// const ueImage = process.env.UERANSIM_IMAGE || "ueransim-base";
const ueImage = process.env.UERANSIM_IMAGE || "ran-node-ue";
const reapIntervalMs = Number(process.env.REAP_INTERVAL_MS || 15000);

// 헬스체크 엔드포인트
app.get("/api/health", (_req, res) => {
  res.json({ status: "ok" });
});

// UE 관리 라우터 등록
app.use(
  "/api/ue",
  createUeRouter(subscribers, {
    docker,
    config: {
      hostDir: configHostDir,
      mountDir: configMountDir,
    },
    network: ranNetwork,
    image: ueImage,
  })
);

const publicDir = path.resolve(__dirname, "..", "public");
app.use(express.static(publicDir));

app.use((req, res) => {
  // fallback for SPA-style routing later
  res.sendFile(path.join(publicDir, "index.html"));
});

/**
 * 주기적으로 ran-ui 관리 하에 있는 컨테이너 중 상태가 exited/dead 인 것들을 정리한다.
 * MongoDB 상태도 함께 초기화해 UI가 최신 정보를 반영하도록 유지한다.
 */
async function reapExitedContainers() {
  try {
    const summaries = await docker.listContainers({
      all: true,
      filters: { label: ["ran-ui=ue"] },
    });

    for (const summary of summaries) {
      if (!["exited", "dead"].includes(summary.State)) {
        continue;
      }
      const imsi = summary.Labels?.["ran-ui.imsi"];
      if (!imsi) {
        continue;
      }
      console.log(`[reap] removing exited UE container ${summary.Names?.[0]} (${summary.Id})`);
      try {
        const container = docker.getContainer(summary.Id);
        await container.remove({ force: true });
      } catch (err) {
        console.error(`[reap] failed to remove container ${summary.Id}`, err);
      }
      await subscribers.updateOne(
        { imsi },
        { $unset: { containerId: "", status: "", configPath: "" } }
      );
    }
  } catch (err) {
    console.error("[reap] failed to enumerate containers", err);
  }
}

setInterval(reapExitedContainers, reapIntervalMs);

app.listen(PORT, HOST, () => {
  console.log(`[ran-ui] listening on http://${HOST}:${PORT}`);
});
