"""
patch_alpamayo_coc.py — alpamayo1_5.py에 max_coc_tokens 지원 추가

목적:
  max_generation_length는 "전체 생성 토큰의 안전망"이지 "CoC 토큰 수 제어"가 아님.
  실제 실험은 CoC 토큰을 N개 생성 후 <|traj_future_start|>를 강제 삽입해야 한다.

  이 스크립트는 alpamayo1_5.py에 다음을 추가한다:
    1. ForceEarlyEOS LogitsProcessor 클래스
    2. sample_trajectories_from_data_with_vlm_rollout의 max_coc_tokens 지원

사용법 (Thor에서 1회만 실행):
  python3 ~/alpamayo1.5/scripts/patch_alpamayo_coc.py

수정 후 원복:
  cp ~/alpamayo1.5/src/alpamayo1_5/models/alpamayo1_5.py.bak \
     ~/alpamayo1.5/src/alpamayo1_5/models/alpamayo1_5.py
"""

import shutil
from pathlib import Path

SRC = Path("/home/ice401/alpamayo1.5/src/alpamayo1_5/models/alpamayo1_5.py")
BAK = SRC.with_suffix(".py.bak")

# ── 백업 ────────────────────────────────────────────────────────────────────────
if not BAK.exists():
    shutil.copy2(SRC, BAK)
    print(f"백업 완료: {BAK}")
else:
    print(f"백업 이미 존재: {BAK}")

# ── 소스 읽기 ────────────────────────────────────────────────────────────────────
src = SRC.read_text(encoding="utf-8")
original_len = len(src)

# ── 패치 1: ForceEarlyEOS 클래스 삽입 ──────────────────────────────────────────
# StopAfterEOS 클래스 정의 바로 뒤에 삽입한다.
# StopAfterEOS가 어디에 정의되어 있는지 찾아서 그 뒤에 추가.

FORCE_EARLY_EOS_CODE = '''

class ForceEarlyEOS(LogitsProcessor):
    """
    CoC 토큰을 정확히 max_coc_tokens개 생성한 뒤
    다음 스텝에서 <|traj_future_start|>(EOS)를 강제 생성한다.

    목적:
      - max_coc_tokens=0: 즉시 EOS → Prefill hidden state만으로 Flow 실행 (진짜 Decode Skip)
      - max_coc_tokens=N: N개 CoC 후 EOS 강제 → N-token CoC 실험

    기존 StopAfterEOS와의 협업:
      ForceEarlyEOS가 EOS logit을 강제 → StopAfterEOS가 EOS 감지 → +1 토큰 후 종료
      → KV cache update까지 정상 완료

    LogitsProcessorList에서 ExpertLogitsProcessor 뒤에 배치해야 한다.
    (마지막으로 실행되어 이전 processor 결과를 덮어씀)
    """

    def __init__(self, eos_token_id: int, max_coc_tokens: int):
        self.eos_token_id = eos_token_id
        self.max_coc = max_coc_tokens
        self._step = 0

    def __call__(
        self,
        input_ids: "torch.LongTensor",
        scores: "torch.FloatTensor",
    ) -> "torch.FloatTensor":
        if self._step >= self.max_coc:
            # EOS 강제: 모든 logit -inf, EOS만 0
            scores = scores.clone()
            scores[:, :] = float("-inf")
            scores[:, self.eos_token_id] = 0.0
        self._step += 1
        return scores

'''

# StopAfterEOS 클래스가 있는 위치 탐색
if "class StopAfterEOS" in src:
    # StopAfterEOS 클래스 블록 끝 찾기 (다음 class 또는 def 앞)
    idx = src.find("class StopAfterEOS")
    # 해당 클래스 블록의 끝: 다음 빈 줄 + 비들여쓰기 라인 탐색
    lines = src.splitlines(keepends=True)
    in_class = False
    insert_after_line = -1
    class_start_line = -1

    for i, line in enumerate(lines):
        if "class StopAfterEOS" in line:
            in_class = True
            class_start_line = i
            continue
        if in_class:
            # 클래스가 끝나는 지점: 들여쓰기 없이 새 class/def/주석 시작
            stripped = line.rstrip()
            if stripped and not stripped.startswith(" ") and not stripped.startswith("\t"):
                insert_after_line = i - 1
                break

    if insert_after_line == -1:
        insert_after_line = len(lines) - 1  # fallback: 파일 끝

    # ForceEarlyEOS를 삽입
    if "class ForceEarlyEOS" not in src:
        lines.insert(insert_after_line, FORCE_EARLY_EOS_CODE)
        src = "".join(lines)
        print("ForceEarlyEOS 클래스 삽입 완료 (StopAfterEOS 뒤)")
    else:
        print("ForceEarlyEOS 이미 존재 — 스킵")
else:
    print("WARNING: StopAfterEOS 클래스를 찾지 못함. 파일 끝에 삽입.")
    if "class ForceEarlyEOS" not in src:
        src = src + FORCE_EARLY_EOS_CODE
        print("ForceEarlyEOS 클래스 파일 끝에 삽입")


# ── 패치 2: sample_trajectories_from_data_with_vlm_rollout에 max_coc_tokens 지원 ──
# 기존 logits_processor 생성 뒤에 max_coc_tokens 처리 코드를 추가한다.

# 타겟: ExpertLogitsProcessor 블록 + vlm.generate 호출 사이
# 기존:
#   logits_processor = LogitsProcessorList([
#       ExpertLogitsProcessor(...)
#   ])
#   vlm_outputs = self.vlm.generate(

OLD_LOGITS_BLOCK = """        logits_processor = LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=self.config.traj_token_start_idx,
                    traj_vocab_size=self.config.traj_vocab_size,
                )
            ]
        )
        vlm_outputs = self.vlm.generate("""

NEW_LOGITS_BLOCK = """        logits_processor = LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=self.config.traj_token_start_idx,
                    traj_vocab_size=self.config.traj_vocab_size,
                )
            ]
        )

        # ── max_coc_tokens: CoC 토큰 수를 정확히 N개로 제한 (실험용) ─────────────
        # max_generation_length 와의 차이:
        #   max_generation_length: 전체 생성 토큰 안전망 (EOS 미생성 시 강제 종료)
        #   max_coc_tokens: EOS(<|traj_future_start|>) 전에 생성할 CoC 토큰 수 제어
        #     N=0: 즉시 EOS → Prefill hidden state만으로 Flow 실행 (Decode Skip)
        #     N=K: K개 CoC 토큰 후 EOS 강제 삽입
        _max_coc_tokens = kwargs.get("max_coc_tokens", None)
        if _max_coc_tokens is not None:
            logits_processor.append(
                ForceEarlyEOS(
                    eos_token_id=eos_token_id,
                    max_coc_tokens=_max_coc_tokens,
                )
            )
            # max_new_tokens를 충분히 확보: CoC(N) + EOS(1) + KV update용(1) + 여유(2)
            generation_config.max_new_tokens = _max_coc_tokens + 4
        # ─────────────────────────────────────────────────────────────────────────

        vlm_outputs = self.vlm.generate("""

count = src.count(OLD_LOGITS_BLOCK)
if count == 0:
    print("ERROR: logits_processor 블록을 찾지 못함. 수동 수정 필요.")
    print("  탐색 대상:")
    print("  " + OLD_LOGITS_BLOCK[:80])
elif count == 1:
    src = src.replace(OLD_LOGITS_BLOCK, NEW_LOGITS_BLOCK, 1)
    print(f"max_coc_tokens 지원 추가 완료 (1개 함수에 적용)")
elif count == 2:
    src = src.replace(OLD_LOGITS_BLOCK, NEW_LOGITS_BLOCK)
    print(f"max_coc_tokens 지원 추가 완료 (2개 함수 모두 적용)")
else:
    print(f"WARNING: {count}개 일치. 첫 번째만 수정.")
    src = src.replace(OLD_LOGITS_BLOCK, NEW_LOGITS_BLOCK, 1)


# ── 저장 ────────────────────────────────────────────────────────────────────────
SRC.write_text(src, encoding="utf-8")
print(f"\n수정 완료: {SRC}")
print(f"  원본 크기: {original_len} chars")
print(f"  수정 크기: {len(src)} chars")
print(f"  추가된 코드: {len(src) - original_len} chars")

# ── 검증 ────────────────────────────────────────────────────────────────────────
import ast
try:
    ast.parse(src)
    print("문법 검사: OK")
except SyntaxError as e:
    print(f"문법 오류! line {e.lineno}: {e.msg}")
    print("백업에서 복원합니다...")
    shutil.copy2(BAK, SRC)
    print("복원 완료.")

print("\n다음 명령으로 패치 확인:")
print("  grep -n 'ForceEarlyEOS\\|max_coc_tokens' "
      "~/alpamayo1.5/src/alpamayo1_5/models/alpamayo1_5.py")
