# 시간축 speculative의 수학 정식화 — 무손실 정리 · 가속 공식 · 시간 일관성 모델

**날짜**: 2026-06-16
**목적**: 우리 측정(안정 16× / 평균 10.8× / corr(편집거리, 수락)=−0.72)을 **닫힌 수식**으로 설명한다.
논문의 이론 골격. 실측 대조는 `260615_01`, `260616_02`, `260616_03`.

---

## 0. 기호

- 목표 모델 $M$, greedy 디코드가 만드는 CoT 토큰열 $y = (y_1,\dots,y_N)$, $y_t=\arg\max M(\cdot\mid y_{<t},x)$.
- draft(추측) $\tilde y$ — 우리 경우 **직전 프레임(100 ms 전)의 greedy CoT**.
- block-verify draft 길이 $\gamma$, 토큰당 수락확률 $\alpha$.

---

## 1. 무손실 정리 (exactness)

**정리.** block-verify greedy speculative의 출력은 동일 forward로 계산한 $M$의 greedy 디코드 $y$와 **정확히
같다.**

**증명(스케치).** block-verify는 입력 $[\,y_{<t}, \tilde y_t, \tilde y_{t+1},\dots]$를 한 번의 forward로 넣고,
각 위치 $i$에서 $p_i=\arg\max M(\cdot\mid y_{<t},\tilde y_{t:i-1})$를 얻는다. 채택 규칙은

$$\text{accept } \tilde y_i \iff \tilde y_i = p_i,\quad\text{앞에서부터 첫 불일치까지.}$$

causal mask에 의해 $p_i$는 $y_{<t}$와 *수락된* draft만 조건으로 한다. 수락된 prefix가 모두 일치했으므로
그 조건은 실제 디코드 경로와 동일 → $p_i$는 그 지점의 greedy 토큰과 같다. 첫 불일치 위치에서는 draft
대신 모델 자신의 $p_i$(bonus)를 취한다. 따라서 어떤 토큰도 "$M$이 그 자리에서 고를 토큰"이 아닌 것은
남지 않는다. 귀납으로 전체 열이 $y$와 일치. ∎

**따름.** 손실은 *구조적으로* 0이다(양자화·근사와 다른 클래스). 유일한 잔차는 batched↔sequential
**부동소수점 동점**(같은 $\arg\max$가 ULP 차이로 갈림)뿐이며 알고리즘 오차가 아니다(`260616_03`).

---

## 2. 가속 공식

한 번의 target forward가 만드는 기대 토큰 수(Leviathan et al. 2023). 토큰당 수락확률이 i.i.d. $\alpha$,
draft 길이 $\gamma$일 때, 채택 prefix 길이 $A=\min\{k:\text{불일치}\}$는 절단 기하분포를 따르고 bonus 1토큰을
더해:

$$\mathbb{E}[\text{tokens per forward}] \;=\; \frac{1-\alpha^{\gamma+1}}{1-\alpha}.$$

**핵심(우리 고유 조건).** draft 생성 비용이 **0이다(zero draft cost)** — 별도 draft 모델의 forward가 필요
없다(직전 프레임 출력을 재사용). 따라서 일반 speculative의 draft 비용 항이 소거되고, 위 기대값이
**그대로 속도이득**이 된다:

$$\boxed{\;\text{speedup} \;=\; \frac{1-\alpha^{\gamma+1}}{1-\alpha}\;}\qquad
\xrightarrow[\;\gamma \ge N\;]{}\qquad \frac{1}{1-\alpha}.$$

CoT 전체를 draft가 덮으면($\gamma\ge N$) 상한은 $1/(1-\alpha)$. 예: $\alpha=0.94\Rightarrow \approx 16\times$ —
실측 안정 프레임 16×와 일치. 평균 $\alpha$가 낮은 혼합에서 평균 10.8×.

> draft 모델을 쓰는 일반 speculative는 $\text{speedup}=\dfrac{1-\alpha^{\gamma+1}}{(1-\alpha)(1+\gamma c)}$
> ($c$=draft/target 비용비)로 $c$에 깎인다. 우리는 $c=0$이라 이 페널티가 없다 — 이것이 무비용 시간 draft
> (zero-cost temporal draft)의 수식상 이점이다.

---

## 3. $\alpha$를 *시간 일관성*으로 모델링 (우리만의 기여)

일반 speculative에서 $\alpha$는 *draft 모델*의 근사 품질에서 온다. 우리에게 $\alpha$는 **프레임 간 추론의
시간 일관성**에서 온다. 직전·현재 CoT의 토큰 편집거리를 $d$라 하면, 평균적으로

$$\alpha \;\approx\; 1-\frac{d}{N},\qquad d=\text{edit}(y^{(f-1)},y^{(f)}).$$

이를 §2에 대입하면 가속이 **장면 변화 $d$의 함수**로 예측된다:

$$\text{speedup}(d)\;\approx\;\frac{1}{1-\alpha}\;=\;\frac{N}{d}\quad(\text{stable, } \gamma\ge N).$$

$d\to0$(장면 안정)이면 speedup→$\infty$(실제론 forward 1회로 하한), $d$가 크면(급변) speedup→1(=baseline,
fallback). 측정한 **corr$(d,\;$1st-block 수락$)=-0.72$** 와 재사용 비율 100%→84%→12%(편집거리 0/1–3/≥4)가
이 단조 관계를 뒷받침한다(`260616_02`).

**$d$가 작은 이유 = 시스템 구조.**  제어 주기 $\Delta t=100$ ms에서 장면(따라서 인과추론 CoT)이 거의
안 변한다. 즉 $\mathbb{E}[d]$는 $\Delta t$의 증가함수이고, 고정 저주기 embodied agent는 구조적으로
$\mathbb{E}[d]\!\approx\!0$ → $\alpha\!\approx\!1$ → 큰 speedup. **이 항(temporal coherence prior)이 generic
speculative엔 없는 우리 모델링의 핵심.**

---

## 4. 정리 — 논문 이론 골격

1. **무손실 정리**: 출력 = greedy(동일 forward). 잔차는 부동소수점 동점뿐.
2. **가속 공식**: $\text{speedup}=\dfrac{1-\alpha^{\gamma+1}}{1-\alpha}\xrightarrow{\gamma\ge N}\dfrac{1}{1-\alpha}$,
   draft 생성 비용 0(zero draft cost)이므로 draft 비용항 소거.
3. **시간 일관성 모델**: $\alpha\approx1-d/N$, $\mathbb{E}[d]$는 $\Delta t$의 증가함수 → 저주기에서 $\alpha\to1$.

→ 측정(16×, 10.8×, $r=-0.72$)을 이론에 못 박고, **③이 우리만의 기여**(embodied 추론의 시간 일관성을
speculative 수락률로 정식화). FastDriveCoT(구조축)·MMSpec(문맥축)과 직교.

### 다음 (이론 강화 후보)
- ✅ **검증 완료**: $\text{speedup}\approx N/d$ 가 실측과 **R²=0.99**(안정 구간 R²=1.00), $\alpha\approx1-d/N$
  corr 0.92 — `260616_06`. 상관(r=−0.72) → 적합된 예측 모델로 승격.
- ✅ **검증 완료**: $\Delta t$ sweep 실측 — $\mathbb{E}[d]$가 Δt에 단조 증가(1.0→2.1→4.4), $\alpha$가
  0.92(100 ms)→0.63(1000 ms)로 감소. 시간 일관성 항 확인, 100 ms가 최대-α 운영점 — `260617_01`.
- TOST 등가성 검정으로 §1 잔차(부동소수점 동점)의 궤적 무영향 정량화.

### 참고
| 항목 | 위치 |
|------|------|
| 무손실 잔차(부동소수점) 진단 | `260616_03`, `umic/scripts/260616_biteq_probe.py` |
| $\alpha$–편집거리 실측 | `260616_02`, `umic/results/260615_draftsrc.csv` |
| 안정 16×·평균치 | `260615_01`, `260616_03` |
