# 무학습 Speculative Decoding이 되는가 — CoT 수락률을 외과의사처럼 측정

**날짜**: 2026-06-14
**원칙**: 구현하기 전에 "이득이 있는가"를 먼저 측정. 추측 금지.

> 한 줄: **단일 문장 안에는 반복이 없어 prompt-lookup은 무용(1.0×). 진짜 반복은 프레임 간(10Hz)에 있고,
> 이전 프레임 CoT를 draft로 쓰면 — 단, sampled decode가 추론을 매 프레임 뒤집어 가린다(2.3×). greedy로
> 그 진동을 없애면 안정 구간에서 17×(평균 13.8×). 출력은 검증으로 항상 동일하다.**

---

## 0. speculative decoding이 뭐고, 왜 무학습으로 출력이 안 변하나

작은 "초안(draft)"이 다음 토큰 몇 개를 미리 제안하면, **큰 모델이 그걸 한 번의 forward로 검증**한다. 맞은
것만 채택하고 틀리면 버린다. → **draft가 틀려도 출력은 절대 안 틀린다**(draft 품질은 속도에만 영향). 그래서
학습 없이도 안전하다. 수학:

  속도 ≈ 토큰수 / forward수 = 평균(수락길이 + 1)

핵심은 **수락길이(draft가 몇 개나 맞히나)를 측정**하는 것. 무학습 draft를 **어디서 가져오나**가 관건이다.

---

## 1. 측정 (연속 6프레임, 100ms 간격, 한 clip)

| draft 소스 | decode 모드 | 평균 속도 |
|---|---|---|
| 같은 문장 안 n-gram (prompt-lookup) | sampled | **1.0× (무용)** |
| 이전 프레임 CoT | sampled (배포 기본) | 2.3× (진동) |
| 이전 프레임 CoT | **greedy** | **13.8× (안정 17×)** |

### 발견 1 — 한 문장 안엔 반복이 없다
CoT는 *"Nudge to the left to clear the construction equipment blocking the right side of our lane"*
같은 **고유한 한 문장**. 내부에 같은 구절이 안 반복되니 prompt-lookup 수락 = 0. → 단일 추론 안에서의
speculative는 운전 CoT엔 무의미.

### 발견 2 — 진짜 반복은 프레임 간에 있다. 그러나 샘플링이 가린다
sampled decode에서 연속 프레임 CoT가 **두 유효 추론 사이를 진동**했다:

![프레임별 CoT: sampled 진동 vs greedy 안정](figures/260614_fig3_spec_oscillation.svg)

- f0,f2,f5 = "앞차 거리유지" / f1,f3,f4 = "공사구간 회피" — 같은 텍스트가 토큰까지 정확히 재현된다.
  장면이 100ms마다 변한 게 아니라, **장면에 유효한 추론이 둘이고 stochastic decode가 그 사이를 샘플링**한 것.
- 그 결과 cross-frame 수락이 이중모드: 같은 추론이면 6.7×, 뒤집히면 1.2× → 평균 2.3×.

### 발견 3 — greedy면 진동이 사라지고 13.8×
temp을 0으로(greedy) 돌리자 **f1~f5가 전부 동일**해졌다. 이제 이전 프레임 CoT가 거의 완벽한 draft:
- 안정 구간(f1→f5): **17토큰 CoT 전체를 단 1 forward로 검증·전부 수락 → 17×**
- 장면이 진짜 바뀐 1회(f0→f1): 1.2×
- 평균 **13.8×**

→ **수락률은 decode 엔트로피에 게이트된다.** FlashDrive가 말한 "reasoning 토큰은 저엔트로피"를 측정으로
정정: 엔트로피는 문장 *내부*가 아니라 **모드 선택(어느 유효 추론을 말할지)** 에 있고, greedy가 그걸 해소한다.

---

## 2. 무엇을 의미하나

- **무학습 cross-frame speculative는 매우 유망**하다. 이전 프레임 CoT를 draft로 쓰는 것 = 10Hz 시간 중복을
  CoT draft에 활용(FlashDrive의 streaming이 KV에 한 걸 CoT draft에). 학습된 drafter(DFlash) 없이 안정 구간
  17×. Decode가 가장 큰 단계(1,330 ms)라 잠재 효과가 adaptive flow보다 크다.
- **출력 정확성**은 검증이 보장: sampled target이면 sampled 출력 그대로(2.3×), greedy target이면 greedy
  출력(13.8×). 각자 자기 target과 동일.

---

## 3. 정직한 단서 (재확인 필요)

1. **게이트 = greedy.** 13.8×는 greedy 전제다. 배포 기본(sampled)은 2.3×. greedy CoT가 주행 품질을 해치지
   않는지 확인 필요 — 단 궤적 다양성(minADE6)은 flow 샘플링에서 오므로 CoT greedy와 분리될 수 있다(측정 요).
2. **최악 케이스는 1×.** 장면이 진짜 바뀌는 프레임은 full decode다. 평균은 빨라도 실시간 worst-case는 변화
   프레임이 지배한다. 그리고 **dynamic long-tail(Alpamayo가 존재하는 이유)일수록 변화가 잦아 speedup이
   낮아진다** — speculative가 가장 약한 곳이 가장 중요한 곳이라는 긴장. 다수 clip·동적 장면으로 분포 측정 필요.
3. 표본은 1 clip·6프레임. 일반화엔 다클립 측정이 선결.

## 4. 부수 발견 (속도와 무관, 안전/품질)

배포 기본(sampled, temp 0.6)에서 모델이 **명시 추론을 100ms마다 두 유효 개념 사이로 뒤집는다**(앞차
거리유지 ↔ 공사 회피). 둘 다 타당하지만, 100ms 진동은 일관성·안전 관점의 관찰거리다. greedy는 안정화한다.

---

## 5. 다음
- 실제 cross-frame speculative decoder 구현(generate 루프: 이전 CoT를 block draft → 1-forward 검증) →
  decode 실측 speedup + 출력 동일성 게이트.
- greedy vs sampled CoT의 궤적 품질(minADE6) 비교로 운영점 결정.

### 참고
| 항목 | 위치 |
|------|------|
| 수락률 측정·코드 | `umic` repo `results/260614_spec_decoding_findings.md`, `scripts/260614_spec_*` |
| FlashDrive 비교 | `docs/2606_2주차/260614_02_FlashDrive_*.md` |
