#!/usr/bin/env bash
set -Eeuo pipefail

# -------------------------------
# 기본 환경변수 (이미 설정돼 있으면 그대로 사용)
# -------------------------------
: "${FG_FOLDER:=/home/main/workspace/k2room2/Baselines/OpenFunGraph/}"
: "${FUNGRAPH3D_ROOT:=/home/main/workspace/k2room2/gpuserver00_storage/CAPA/FunGraph3D/}"
: "${FUNGRAPH3D_CONFIG_PATH:=${FG_FOLDER%/}/openfungraph/dataset/dataconfigs/fungraph3d/fungraph3d.yaml}"
export FG_FOLDER FUNGRAPH3D_ROOT FUNGRAPH3D_CONFIG_PATH

# (호환용) detection 스크립트가 SCENEFUN3D_ROOT를 기대할 수도 있어 둘 다 export
export SCENEFUN3D_ROOT="${FUNGRAPH3D_ROOT%/}/"

# 실행할 대상 스크립트
TARGET_SCRIPT="${FG_FOLDER%/}/openfungraph/scenegraph/detection_fungraph3d.sh"

# -------------------------------
# 선택 파라미터
# 사용: ./run_all_fungraph3d.sh [scene_glob] [video_glob]
# 예  : ./run_all_fungraph3d.sh "10kitchen" "video0"
# 기본: scene_glob="*" , video_glob="video*"
# -------------------------------
SCENE_GLOB="${1:-*}"
VIDEO_GLOB="${2:-video*}"

# 경로 검증
BASE="${FUNGRAPH3D_ROOT%/}"
if [[ ! -f "$TARGET_SCRIPT" ]]; then
  echo "[ERROR] target script not found: $TARGET_SCRIPT" >&2
  exit 1
fi
if [[ ! -d "$BASE" ]]; then
  echo "[ERROR] FUNGRAPH3D_ROOT not found: $BASE" >&2
  exit 1
fi

# 로그 디렉터리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs_fungraph3d"
mkdir -p "$LOG_DIR"
FAILED_LIST="${LOG_DIR}/failed_scenes.txt"
: > "$FAILED_LIST"

echo "[INFO] FG_FOLDER        = $FG_FOLDER"
echo "[INFO] TARGET_SCRIPT    = $TARGET_SCRIPT"
echo "[INFO] FUNGRAPH3D_ROOT  = $FUNGRAPH3D_ROOT"
echo "[INFO] scene_glob       = $SCENE_GLOB"
echo "[INFO] video_glob       = $VIDEO_GLOB"
echo "[INFO] Logs             = $LOG_DIR"
echo

TOTAL=0; OK=0; FAIL=0

PROCESSED=(0kitchen 10kitchen 11bedroom 12kitchen) # 이미 완료된 scene은 스킵

# <scene>/<video*> 디렉터리만 순회
while IFS= read -r -d '' SUBDIR; do
  REL="${SUBDIR#${BASE}/}"       # 예: 10kitchen/video0
  export SCENE_NAME="${REL}/"    # 요청 형식대로 끝에 '/' 포함

  
  SCENE="${REL%%/*}" 
  # SCENE이 처리 목록에 '있으면' 스킵
  if [[ " ${PROCESSED[*]} " == *" ${SCENE} "* ]]; then
    echo "[SKIP processed] scene=$SCENE path=$SUBDIR"
    continue
  fi


  SAFE_NAME="${REL//\//_}"       # 로그 파일명
  OUT_LOG="${LOG_DIR}/${SAFE_NAME}.out"
  ERR_LOG="${LOG_DIR}/${SAFE_NAME}.err"

  ((TOTAL++)) || true
  echo "=============================="
  echo "[START] ${SCENE_NAME}  (abs: ${SUBDIR})"
  echo "------------------------------"

  if bash "$TARGET_SCRIPT" >"$OUT_LOG" 2>"$ERR_LOG"; then
    echo "[OK]    ${SCENE_NAME}"
    ((OK++)) || true
  else
    echo "[FAIL]  ${SCENE_NAME}  path=${SUBDIR}"
    echo "$SUBDIR" >> "$FAILED_LIST"
    tail -n 5 "$ERR_LOG" 2>/dev/null || true
    ((FAIL++)) || true
  fi
  echo
done < <(find "$BASE" -mindepth 2 -maxdepth 2 -type d -path "$BASE/$SCENE_GLOB/$VIDEO_GLOB" -print0 | sort -z)

echo "=============================="
echo "[DONE] processed under: $BASE"
echo "[STATS] total=$TOTAL  ok=$OK  fail=$FAIL"
if [[ -s "$FAILED_LIST" ]]; then
  echo "[INFO] Failed scenes listed at: $FAILED_LIST"
else
  echo "[INFO] All scenes succeeded."
fi
