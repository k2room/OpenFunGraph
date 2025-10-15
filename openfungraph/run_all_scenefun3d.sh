#!/usr/bin/env bash
set -Eeuo pipefail

# -------------------------------
# 기본 환경변수 (이미 설정돼 있으면 그대로 사용)
# -------------------------------
: "${FG_FOLDER:=/home/main/workspace/k2room2/Baselines/OpenFunGraph/}"
: "${SCENEFUN3D_ROOT:=/home/main/workspace/k2room2/gpuserver00_storage/CAPA/SceneFun3D_Graph/test/}"
: "${SCENEFUN3D_CONFIG_PATH:=${FG_FOLDER%/}/openfungraph/dataset/dataconfigs/scenefun3d/scenefun3d.yaml}"
export FG_FOLDER SCENEFUN3D_CONFIG_PATH

# 실행할 대상 스크립트 (요청하신 경로)
TARGET_SCRIPT="${FG_FOLDER%/}/openfungraph/scenegraph/detection_scenefun3d.sh"

# -------------------------------
# 파라미터: SPLIT = test | dev
# 사용: ./run_all_scenefun3d.sh [test|dev]
# 기본값: test
# -------------------------------
SPLIT="${1:-test}"
if [[ "$SPLIT" != "test" && "$SPLIT" != "dev" ]]; then
  echo "[USAGE] $0 [test|dev]" >&2
  exit 2
fi

# SCENEFUN3D_ROOT가 /.../SceneFun3D_Graph 또는 /.../SceneFun3D_Graph/{test,dev} 일 수 있음
ROOT_TRIM="${SCENEFUN3D_ROOT%/}"
BN="$(basename "$ROOT_TRIM")"
if [[ "$BN" == "test" || "$BN" == "dev" ]]; then
  ROOT_PREFIX="$(dirname "$ROOT_TRIM")"
else
  ROOT_PREFIX="$ROOT_TRIM"
fi

BASE="${ROOT_PREFIX%/}/${SPLIT}"
BASE="${BASE%/}"
export SCENEFUN3D_ROOT="${BASE}/"   # 실행 대상 스크립트가 참조하도록 export

# 경로 검증
if [[ ! -f "$TARGET_SCRIPT" ]]; then
  echo "[ERROR] target script not found: $TARGET_SCRIPT" >&2
  exit 1
fi
if [[ ! -d "$BASE" ]]; then
  echo "[ERROR] split dir not found: $BASE" >&2
  exit 1
fi

# 로그 디렉터리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs_${SPLIT}"
mkdir -p "$LOG_DIR"
FAILED_LIST="${LOG_DIR}/failed_scenes.txt"
: > "$FAILED_LIST"

echo "[INFO] FG_FOLDER         = $FG_FOLDER"
echo "[INFO] TARGET_SCRIPT     = $TARGET_SCRIPT"
echo "[INFO] SCENEFUN3D_ROOT   = $SCENEFUN3D_ROOT"
echo "[INFO] SPLIT             = $SPLIT"
echo "[INFO] Logs              = $LOG_DIR"
echo

# test scene: (421380 422391 422813 422826 460417 460419 466183 466192 466803 467293 468076 469011)
PROCESSED=(421380 422391 422813 422826 460417 460419 466183 466192) # 이미 완료된 scene은 스킵

# test|dev/<scene>/<video>/ 형태 순회
while IFS= read -r -d '' SUBDIR; do
  REL="${SUBDIR#${BASE}/}"          # 예: 469011/45663175
  export SCENE_NAME="${REL}/"       # 요청 형식대로 끝에 '/' 포함
  
  
  SCENE="${REL%%/*}" 
  # SCENE이 처리 목록에 '있으면' 스킵
  if [[ " ${PROCESSED[*]} " == *" ${SCENE} "* ]]; then
    echo "[SKIP processed] scene=$SCENE path=$SUBDIR"
    continue
  fi


  SAFE_NAME="${REL//\//_}"          # 로그 파일명
  OUT_LOG="${LOG_DIR}/${SAFE_NAME}.out"
  ERR_LOG="${LOG_DIR}/${SAFE_NAME}.err"

  echo "=============================="
  echo "[START] ${SCENE_NAME}  (abs: ${SUBDIR})"
  echo "------------------------------"

  if bash "$TARGET_SCRIPT" >"$OUT_LOG" 2>"$ERR_LOG"; then
    echo "[OK]    ${SCENE_NAME}"
  else
    echo "[FAIL]  ${SCENE_NAME}  path=${SUBDIR}"
    echo "$SUBDIR" >> "$FAILED_LIST"
    tail -n 5 "$ERR_LOG" 2>/dev/null || true
  fi
  echo
done < <(find "$BASE" -mindepth 2 -maxdepth 2 -type d -print0 | sort -z)

echo "=============================="
echo "[DONE] processed all under: $BASE"
if [[ -s "$FAILED_LIST" ]]; then
  echo "[INFO] Failed scenes listed at: $FAILED_LIST"
else
  echo "[INFO] All scenes succeeded."
fi
