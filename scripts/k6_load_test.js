/**
 * k6 WebSocket load test for DecisionDesk.
 *
 * Measures P95 end-to-end latency across N concurrent review sessions.
 *
 * Usage:
 *   k6 run --vus 10 --duration 3m scripts/k6_load_test.js
 *   k6 run --vus 10 --duration 3m -e HOST=http://your-hetzner-ip:8080 scripts/k6_load_test.js
 */

import ws from "k6/ws";
import { check, sleep } from "k6";
import { Trend, Counter } from "k6/metrics";

const reviewDuration = new Trend("review_duration_ms", true);
const reviewsCompleted = new Counter("reviews_completed");
const reviewsFailed = new Counter("reviews_failed");

export const options = {
  scenarios: {
    reviews: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: "30s", target: 3 },   // ramp up
        { duration: "5m",  target: 3 },   // hold at 3 VUs (~12 concurrent API calls vs 20)
      ],
      gracefulRampDown: "2m",             // let in-flight reviews finish
    },
  },
  thresholds: {
    review_duration_ms: ["p(95)<60000"],
    reviews_failed: ["count<5"],
  },
};

const HOST = __ENV.HOST || "ws://localhost:8080";

const DECISIONS = [
  { question: "Should we use PostgreSQL or DynamoDB?", option_a: "PostgreSQL", option_b: "DynamoDB" },
  { question: "Should we use Redis or Memcached as our cache?", option_a: "Redis", option_b: "Memcached" },
  { question: "Should we build microservices or a monolith?", option_a: "Microservices", option_b: "Monolith" },
  { question: "Should we use Kafka or RabbitMQ for event streaming?", option_a: "Kafka", option_b: "RabbitMQ" },
  { question: "Should we use REST or GraphQL for our API?", option_a: "REST", option_b: "GraphQL" },
];

export default function () {
  const decision = DECISIONS[Math.floor(Math.random() * DECISIONS.length)];
  const url = `${HOST.replace(/^http/, "ws")}/ws/review`;

  const t0 = Date.now();
  let completed = false;
  let connected = false;

  const res = ws.connect(url, {}, (socket) => {
    socket.on("open", () => {
      connected = true;
      socket.send(
        JSON.stringify({
          type: "start",
          question: decision.question,
          option_a: decision.option_a,
          option_b: decision.option_b,
          constraints: { team_size: "medium", traffic: "medium", budget: "medium" },
          force_approve: false,
        })
      );
    });

    socket.on("message", (data) => {
      const event = JSON.parse(data);
      if (event.type === "review_complete") {
        reviewDuration.add(Date.now() - t0);
        reviewsCompleted.add(1);
        completed = true;
        socket.close();
      } else if (event.type === "error") {
        reviewsFailed.add(1);
        socket.close();
      }
    });

    socket.on("error", () => {
      reviewsFailed.add(1);
    });

    // Hard timeout: close after 120s if no review_complete
    socket.setTimeout(() => {
      if (!completed) {
        reviewsFailed.add(1);
        socket.close();
      }
    }, 300000);
  });

  check(connected, { "WS connected": (v) => v === true });
  sleep(1);
}
