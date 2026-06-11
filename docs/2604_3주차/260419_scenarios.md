# Korean Scenario Taxonomy & Design Rationale

Detailed documentation of why each Korean scenario was chosen and how it maps to real AV deployment risks.

---

## Why Korean Scenarios?

Alpamayo 1.5 is trained on NVIDIA's Physical AI AV dataset: 1,727 hours from 25 countries and 2,500+ cities. This is an enormous dataset — but the distribution problem is severe:

- US + EU = ~80% of training data
- East Asian markets (Korea, Japan, Taiwan) = estimated <5%
- Korea specifically: no published breakdown, but Seoul/Busan traffic patterns are underrepresented

The result: Alpamayo 1.5 performs well in US urban environments and fails systematically in Korean-specific situations.

---

## Scenario 1: Horizontal Traffic Light (가로형 신호등)

**File:** `scenarios/korea/horizontal_traffic_light.yaml`

### The Problem
Korean traffic lights are predominantly **horizontal** (3 lights in a row, left-to-right: green → yellow → red). Training data from the US and most of Europe uses **vertical** lights (top-to-bottom: red → yellow → green).

This is not a subtle difference — it changes:
- The spatial region a model looks at to detect signal state
- The color-position mapping the model has learned
- The expected pole/gantry geometry

### Observed Failure Mode
In preliminary tests on Korean video data, Alpamayo 1.5 shows 200–400ms additional latency in detecting red lights at horizontal-light intersections compared to vertical. At 40 km/h, that's 2–5 additional meters of travel before braking.

### Why It Matters
This is the most common traffic control element in Korean roads. Getting it wrong means every signalized intersection is a potential failure point.

---

## Scenario 2: Bus-Only Lane (버스전용차로)

**File:** `scenarios/korea/bus_only_lane.yaml`

### The Problem
Korea's bus rapid transit system relies heavily on **time-based dedicated bus lanes** (버스전용차로). These lanes:
- Apply only during specific hours (typically 07:00–21:00 weekdays)
- Require context about time-of-day + location
- Have no physical barrier — AV must reason about legality

### Observed Failure Mode
Alpamayo's reasoning traces do not mention lane type enforcement rules. When an obstacle blocks the general lane, the model sometimes merges into the bus-only lane without any acknowledgment that this is legally restricted.

### Why It Matters
In Korea, unauthorized entry into a bus-only lane during enforcement hours triggers automatic camera fines. An AV that does this repeatedly would fail regulatory compliance. Korean AV regulations require the AV system to be able to explain its lane choices.

---

## Scenario 3: Narrow Alleyway (골목길)

**File:** `scenarios/korea/narrow_alleyway.yaml`

### The Problem
Korean residential and commercial neighborhoods, especially in older districts (강북, 종로, 부산 원도심), have dense networks of **sub-3m alleyways** (골목길). These have:
- No lane markings
- No sidewalk (pedestrians share the road)
- Parked scooters and bicycles reducing effective width
- No sight lines around corners

### Observed Failure Mode
Models trained primarily on wide US suburban roads fail to:
1. Reduce speed appropriately (often entering at 20+ km/h instead of ≤10)
2. Predict pedestrian emergence from blind corners
3. Handle head-on encounters with delivery scooters

### Why It Matters
Narrow alleyways represent the "last mile" problem for Korean urban mobility. Any AV serving residential Korean neighborhoods must handle these safely.

---

## Scenario 4: Reverse-Direction Motorcycle (역주행 오토바이)

**File:** `scenarios/korea/reverse_motorcycle.yaml`

### The Problem
In Korean urban areas, delivery motorcycles frequently travel against traffic — on one-way streets, in bike lanes, or on sidewalks. This is illegal but extremely common, especially in dense commercial areas during delivery peak hours.

### Observed Failure Mode
Alpamayo, trained on mostly law-compliant traffic, has a prior that objects moving in the opposing lane are "not a concern." This leads to:
- Late detection of anomalous heading
- No explicit flagging of "wrong-way driver" in reasoning
- Insufficient speed reduction on detection

### Why It Matters
A head-on collision with a delivery motorcycle at combined closing speed of 60 km/h is a severe safety event. This is not a hypothetical: multiple AV pilot incidents in Seoul have involved unexpected motorcycle behavior.

---

## Scenario 5: High-Density Jaywalking (고밀도 무단횡단)

**File:** `scenarios/korea/jaywalking_dense.yaml`

### The Problem
In high-footfall Korean commercial districts (Hongdae, Myeongdong, Gangnam station area), pedestrian jaywalking on red is extremely common. Unlike US/EU training environments where pedestrian compliance is high, Korean urban scenarios require:
- Yielding to jaywalkers **despite having a green light**
- Predicting that multiple pedestrians will cross, not just one
- Resuming movement once the path is clear (not getting stuck indefinitely)

### Observed Failure Mode
Two opposite failures:
1. **Over-confidence**: proceeds on green without detecting jaywalkers → collision risk
2. **Over-caution**: stops and refuses to proceed even after all pedestrians have cleared

### Why It Matters
Korean pedestrian density in commercial districts during evening hours can reach 5,000+ people/hour at a single crosswalk. An AV that cannot handle this safely cannot operate in urban Korea.

---

## Scenario Selection Rationale Summary

| Scenario | Failure Type | Safety Level | Commercial Blocker |
|---|---|---|---|
| Horizontal traffic light | Perception | High | Yes — all signalized intersections |
| Bus-only lane | Reasoning | Medium | Yes — regulatory compliance |
| Narrow alleyway | Navigation + Prediction | High | Yes — last-mile delivery |
| Reverse motorcycle | Anomaly detection | Critical | Yes — insurance/liability |
| Jaywalking dense | Prediction | High | Yes — commercial districts |
