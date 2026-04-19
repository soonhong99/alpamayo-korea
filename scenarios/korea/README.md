# Korean Scenario Design Rationale

This directory contains AlpaSim scenario definitions for Korean long-tail driving scenarios. Each scenario targets a specific failure mode of global AV models in Korean road environments.

---

## Scenario Overview

| File | ID | Priority | Key Challenge |
|---|---|---|---|
| `horizontal_traffic_light.yaml` | `horizontal_traffic_light_v1` | High | Perception: horizontal vs vertical light orientation |
| `bus_only_lane.yaml` | `bus_only_lane_v1` | High | Rule reasoning: time-based enforcement logic |
| `narrow_alleyway.yaml` | `narrow_alleyway_v1` | High | Navigation: sub-3m shared space + sudden pedestrians |
| `reverse_motorcycle.yaml` | `reverse_motorcycle_v1` | Medium | Safety: anomalous agent detection + emergency avoidance |
| `jaywalking_dense.yaml` | `jaywalking_dense_v1` | High | Prediction: non-compliant pedestrians despite green signal |

---

## Why These Five?

These scenarios were selected based on three criteria:

1. **Systematic Alpamayo baseline failure** — the global model consistently fails or produces poor reasoning in these situations
2. **Unique to Korean road context** — not well-represented in US/EU training data
3. **Commercially significant** — a failure in any of these scenarios would block AV deployment in Korea

### Evidence of baseline failure

| Scenario | Expected Alpamayo 1.5 Failure Mode |
|---|---|
| Horizontal traffic light | Delayed red-light detection; reasoning doesn't mention orientation |
| Bus-only lane | No awareness of time-based enforcement; sometimes illegally enters lane |
| Narrow alleyway | Overshoots safe speed; fails to yield to sudden pedestrians |
| Reverse motorcycle | Late detection; no explicit anomaly flagging in reasoning |
| Jaywalking dense | Proceeds on green without yielding; treats jaywalkers as static |

---

## YAML Schema Reference

Each scenario YAML must include:

```yaml
scenario:
  id: unique_scenario_id
  name: "Display name (Korean preferred)"
  version: "1.0"
  author: "alpamayo-korea"

  scene:
    source: synthetic | nurec      # nurec = from NVIDIA Physical AI dataset
    environment: { ... }
    road_layout: { ... }

  ego:
    start_position: [x, y, z]
    start_velocity: float          # m/s
    goal_position: [x, y, z]

  agents:
    - type: vehicle | pedestrian | motorcycle | bus
      start: [x, y, heading_rad]
      behavior: behavior_id

  events:
    - at_time: float
      trigger: trigger_id
      params: { ... }

  success_criteria:
    - type: criterion_id
      # criterion-specific params

  failure_conditions:
    - type: condition_id

  metrics:
    - name: metric_name
      description: "..."
      type: float | boolean        # default: float
```

---

## Adding a New Scenario

1. Create `scenarios/korea/your_scenario.yaml` following the schema above
2. Register it in `configs/alpasim_korea.yaml` under `korea_scenarios:`
3. Add scenario weight in `configs/finetune_config.yaml` under `training.scenario_weights`
4. Update this README with a row in the table above
5. Add Korean reasoning keywords in `configs/finetune_config.yaml` under `reward.reasoning_keywords`

---

## Future Scenarios (Planned)

- `school_zone_speed.yaml` — 스쿨존 30km/h 제한 및 어린이 보호구역 인식
- `highway_merge_korean.yaml` — 고속도로 진입로 합류 (한국 진출입로 구조)
- `rain_night_urban.yaml` — 야간 우천 시 신호 인식 저하
- `construction_zone.yaml` — 임시 차선 변경 구간 (공사 중 임시 신호수)
- `parking_lot_exit.yaml` — 지하주차장 출구 무단횡단 보행자
