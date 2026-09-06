#!/bin/bash

set -u  # 禁止未定义变量
# 不使用 set -e，允许单个任务失败后继续跑后续任务

# =========================================================
# 基本参数
# =========================================================
DEFAULT_GAMMA=0.98
DEFAULT_ALPHA=1
DEFAULT_SIGHT=10
DEFAULT_FIXED_INTENSITY=0.5

TRAIN_START=30000
TRAIN_EPISODES=20
TRAIN_TIME=100

TEST_START=30000
TEST_TIME=100

MAX_JOBS=3
SEED_REPS=10  # 【核心】每个 test 配置重复实验次数

# =========================================================
# 实验矩阵
# =========================================================
TH_TYPES=(dos sybil)
# TH_FIXED_ATTACKS=(1 0.8 0.6 0.4 0.2)
TH_ADAPTIVE_THRESHOLDS=(0.5)

# RL_FIXED_TYPES=(dos sybil)
RL_FIXED_ATTACKS=(1 0.8 0.6 0.4 0.2)

RL_ADAPTIVE_TYPES=(dos sybil)
RL_ADAPTIVE_ALPHAS=(0 0.2 0.5 1 1.5 2)
RL_ADAPTIVE_SIGHTS=(1 10 20)

CB_ADAPTIVE_TYPES=(dos sybil)
CB_ADAPTIVE_ALPHAS=(0 0.2 0.5 1 1.5 2)
CB_ADAPTIVE_SIGHTS=(1 10 20)

# =========================================================
# 日志系统
# =========================================================
LOG_ROOT="logs/exp_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
CFG_ROOT="$LOG_ROOT/configs"
mkdir -p "$CFG_ROOT"

SUCCESS_LOG="$LOG_ROOT/success_tasks.txt"
FAILED_LOG="$LOG_ROOT/failed_tasks.txt"
WARN_LOG="$LOG_ROOT/warn.txt"
touch "$SUCCESS_LOG" "$FAILED_LOG" "$WARN_LOG"

# =========================================================
# 工具函数
# =========================================================
declare -a BG_PIDS=()

norm_num() {
    echo "$1" | sed 's/-/m/g; s/\./p/g'
}

count_active_jobs() {
    local n=0 pid
    for pid in "${BG_PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null && n=$((n + 1))
    done
    echo "$n"
}

wait_for_slot() {
    while [ "$(count_active_jobs)" -ge "$MAX_JOBS" ]; do sleep 1; done
}

wait_all_jobs() {
    local status=0 pid
    for pid in "${BG_PIDS[@]}"; do wait "$pid" || status=1; done
    BG_PIDS=()
    return $status
}

launch_job() {
    local task_name="$1"; shift
    local log_file="$LOG_ROOT/${task_name}.log"
    wait_for_slot
    
    echo "[🚀 QUEUE] $task_name" >&2
    
    (
        echo "[▶️  START] $task_name @ $(date +%H:%M:%S)" | tee -a "$log_file" >&2
        "$@" 2>>"$log_file"
        local status=$?
        
        if [[ $status -eq 0 ]]; then
            echo "[✅ DONE] $task_name @ $(date +%H:%M:%S)" | tee -a "$log_file" >&2
            printf "%s\n" "$task_name" >> "$SUCCESS_LOG"
        else
            echo "[❌ FAIL] $task_name @ $(date +%H:%M:%S) (exit $status)" | tee -a "$log_file" >&2
            printf "%s\n" "$task_name" >> "$FAILED_LOG"
        fi
        exit $status
    ) >>"$log_file" 2>&1 &
    
    BG_PIDS+=("$!")
}

prepare_cfg() {
    local base_cfg="$1"; local out_cfg="$2"; shift 2
    mkdir -p "$(dirname "$out_cfg")"
    cp "$base_cfg" "$out_cfg"
    for expr in "$@"; do yq eval "$expr" -i "$out_cfg"; done
}

make_model_path() {
    echo "outputs/gnn_outputs/single_${1}_gnn"
}

# =========================================================
# 路径生成函数 (严格对齐新规范)
# =========================================================
# train: 含 tr/tt/ep, 不含 ts/t/seed
# test:  含 ts/t/seed, 不含 tr/tt/ep

# --- Threshold (纯 test, 无训练) ---

th_fixed_save_name_test() {
    local type="$1" atk="$2" seed="${3:-}"
    local base="test/test_th_fixed_${type}_a$(norm_num "$atk")_thr0p5_ts${TEST_START}_t${TEST_TIME}"
    [[ -n "$seed" ]] && echo "${base}_seed${seed}" || echo "$base"
}

th_adapt_save_name_test() {
    local type="$1" thr="$2" seed="${3:-}"
    local base="test/test_th_adaptive_${type}_thr$(norm_num "$thr")_ts${TEST_START}_t${TEST_TIME}"
    [[ -n "$seed" ]] && echo "${base}_seed${seed}" || echo "$base"
}

# --- RL Fixed ---

rl_fixed_save_name_train() {
    local type="$1" atk="$2" alpha="$3" sight="$4"
    echo "train/train_rl_fixed_${type}_a$(norm_num "$atk")_al$(norm_num "$alpha")_s${sight}_tr${TRAIN_START}_tt${TRAIN_TIME}_ep${TRAIN_EPISODES}"
}

rl_fixed_save_name_test() {
    local type="$1" atk="$2" alpha="$3" sight="$4" seed="${5:-}"
    local base="test/test_rl_fixed_${type}_a$(norm_num "$atk")_al$(norm_num "$alpha")_s${sight}_ts${TEST_START}_t${TEST_TIME}"
    [[ -n "$seed" ]] && echo "${base}_seed${seed}" || echo "$base"
}

# --- RL Adaptive ---

rl_adapt_save_name_train() {
    local type="$1" alpha="$2" sight="$3"
    echo "train/train_rl_adaptive_${type}_al$(norm_num "$alpha")_s$(norm_num "$sight")_tr${TRAIN_START}_tt${TRAIN_TIME}_ep${TRAIN_EPISODES}"
}

rl_adapt_save_name_test() {
    local type="$1" alpha="$2" sight="$3" seed="${4:-}"
    local base="test/test_rl_adaptive_${type}_al$(norm_num "$alpha")_s$(norm_num "$sight")_ts${TEST_START}_t${TEST_TIME}"
    [[ -n "$seed" ]] && echo "${base}_seed${seed}" || echo "$base"
}

# --- CB Adaptive ---

cb_adapt_save_name_train() {
    local type="$1" alpha="$2" sight="$3"
    echo "train/train_cb_adaptive_${type}_al$(norm_num "$alpha")_s$(norm_num "$sight")_tr${TRAIN_START}_tt${TRAIN_TIME}_ep${TRAIN_EPISODES}"
}

cb_adapt_save_name_test() {
    local type="$1" alpha="$2" sight="$3" seed="${4:-}"
    local base="test/test_cb_adaptive_${type}_al$(norm_num "$alpha")_s$(norm_num "$sight")_ts${TEST_START}_t${TEST_TIME}"
    [[ -n "$seed" ]] && echo "${base}_seed${seed}" || echo "$base"
}

# =========================================================
# 实验单元函数
# =========================================================

# --- Threshold (纯 test) ---

run_th_fixed_unit() {
    local type="$1" atk="$2" seed="$3"
    local model_path=$(make_model_path "$type")
    local save_name=$(th_fixed_save_name_test "$type" "$atk" "$seed")
    local cfg="$CFG_ROOT/${save_name//\//_}.yaml"

    prepare_cfg "config/test_th.yaml" "$cfg" \
        ".mode = \"test\"" ".test_mode = \"threshold\"" \
        ".save_name = \"${save_name}\"" \
        ".env.attack_mode = \"fixed\"" ".env.fixed_intensity = ${atk}" \
        ".env.alpha = ${DEFAULT_ALPHA}" ".env.sight = ${DEFAULT_SIGHT}" \
        ".env.start_time = ${TEST_START}" ".env.max_step = ${TEST_TIME}" \
        ".env.data_path = \"${type}.csv\"" \
        ".env.scaler_path = \"${model_path}/scaler.pkl\"" \
        ".env.gnn_path = \"${model_path}/best_model.pt\"" \
        ".threshold.risk_threshold = 0.5" ".seed = ${seed}"

    python -m code.train.simulation --config "$cfg"
}

run_th_adaptive_unit() {
    local type="$1" thr="$2" seed="$3"
    local model_path=$(make_model_path "$type")
    local save_name=$(th_adapt_save_name_test "$type" "$thr" "$seed")
    local cfg="$CFG_ROOT/${save_name//\//_}.yaml"

    prepare_cfg "config/test_th.yaml" "$cfg" \
        ".mode = \"test\"" ".test_mode = \"threshold\"" \
        ".save_name = \"${save_name}\"" \
        ".env.attack_mode = \"adaptive\"" ".env.fixed_intensity = ${DEFAULT_FIXED_INTENSITY}" \
        ".env.alpha = ${DEFAULT_ALPHA}" ".env.sight = ${DEFAULT_SIGHT}" \
        ".env.start_time = ${TEST_START}" ".env.max_step = ${TEST_TIME}" \
        ".env.data_path = \"${type}.csv\"" \
        ".env.scaler_path = \"${model_path}/scaler.pkl\"" \
        ".env.gnn_path = \"${model_path}/best_model.pt\"" \
        ".threshold.risk_threshold = ${thr}" ".seed = ${seed}"

    python -m code.train.simulation --config "$cfg"
}

# --- RL Fixed: Train + Test ---

run_rl_fixed_train_unit() {
    local type="$1" atk="$2"
    local model_path=$(make_model_path "$type")
    local save_name=$(rl_fixed_save_name_train "$type" "$atk" "$DEFAULT_ALPHA" "$DEFAULT_SIGHT")
    local cfg="$CFG_ROOT/${save_name//\//_}.yaml"

    prepare_cfg "config/train_rl.yaml" "$cfg" \
        ".mode = \"train\"" ".train_mode = \"rl\"" \
        ".save_name = \"${save_name}\"" \
        ".env.attack_mode = \"fixed\"" ".env.fixed_intensity = ${atk}" \
        ".env.alpha = ${DEFAULT_ALPHA}" ".env.sight = ${DEFAULT_SIGHT}" \
        ".env.start_time = ${TRAIN_START}" ".env.max_step = ${TRAIN_TIME}" \
        ".train_episodes = ${TRAIN_EPISODES}" ".episodes = ${TRAIN_EPISODES}" \
        ".agent.gamma = ${DEFAULT_GAMMA}" \
        ".env.data_path = \"${type}.csv\"" \
        ".env.scaler_path = \"${model_path}/scaler.pkl\"" \
        ".env.gnn_path = \"${model_path}/best_model.pt\""

    python -m code.train.simulation --config "$cfg"
}

run_rl_fixed_test_unit() {
    local type="$1" atk="$2" seed="$3"
    local model_path=$(make_model_path "$type")
    
    local train_dir="outputs/simulation/$(rl_fixed_save_name_train "$type" "$atk" "$DEFAULT_ALPHA" "$DEFAULT_SIGHT")"
    local test_ckpt="${train_dir}/final_agent.pt"
    
    local save_name=$(rl_fixed_save_name_test "$type" "$atk" "$DEFAULT_ALPHA" "$DEFAULT_SIGHT" "$seed")
    local cfg="$CFG_ROOT/${save_name//\//_}.yaml"

    prepare_cfg "config/test_rl.yaml" "$cfg" \
        ".mode = \"test\"" ".test_mode = \"rl\"" \
        ".save_name = \"${save_name}\"" \
        ".env.attack_mode = \"fixed\"" ".env.fixed_intensity = ${atk}" \
        ".env.alpha = ${DEFAULT_ALPHA}" ".env.sight = ${DEFAULT_SIGHT}" \
        ".env.start_time = ${TEST_START}" ".env.max_step = ${TEST_TIME}" \
        ".agent.checkpoint_path = \"${test_ckpt}\"" \
        ".env.data_path = \"${type}.csv\"" \
        ".env.scaler_path = \"${model_path}/scaler.pkl\"" \
        ".env.gnn_path = \"${model_path}/best_model.pt\"" \
        ".seed = ${seed}"

    python -m code.train.simulation --config "$cfg"
}

# --- RL Adaptive: Train + Test ---

run_rl_adaptive_train_unit() {
    local type="$1" alpha="$2" sight="$3"
    local model_path=$(make_model_path "$type")
    local save_name=$(rl_adapt_save_name_train "$type" "$alpha" "$sight")
    local cfg="$CFG_ROOT/${save_name//\//_}.yaml"

    prepare_cfg "config/train_rl.yaml" "$cfg" \
        ".mode = \"train\"" ".train_mode = \"rl\"" \
        ".save_name = \"${save_name}\"" \
        ".env.attack_mode = \"adaptive\"" ".env.fixed_intensity = ${DEFAULT_FIXED_INTENSITY}" \
        ".env.alpha = ${alpha}" ".env.sight = ${sight}" \
        ".env.start_time = ${TRAIN_START}" ".env.max_step = ${TRAIN_TIME}" \
        ".train_episodes = ${TRAIN_EPISODES}" ".episodes = ${TRAIN_EPISODES}" \
        ".agent.gamma = ${DEFAULT_GAMMA}" \
        ".env.data_path = \"${type}.csv\"" \
        ".env.scaler_path = \"${model_path}/scaler.pkl\"" \
        ".env.gnn_path = \"${model_path}/best_model.pt\""

    python -m code.train.simulation --config "$cfg"
}

run_rl_adaptive_test_unit() {
    local type="$1" alpha="$2" sight="$3" seed="$4"
    local model_path=$(make_model_path "$type")
    
    local train_dir="outputs/simulation/$(rl_adapt_save_name_train "$type" "$alpha" "$sight")"
    local test_ckpt="${train_dir}/final_agent.pt"
    
    local save_name=$(rl_adapt_save_name_test "$type" "$alpha" "$sight" "$seed")
    local cfg="$CFG_ROOT/${save_name//\//_}.yaml"

    prepare_cfg "config/test_rl.yaml" "$cfg" \
        ".mode = \"test\"" ".test_mode = \"rl\"" \
        ".save_name = \"${save_name}\"" \
        ".env.attack_mode = \"adaptive\"" ".env.fixed_intensity = ${DEFAULT_FIXED_INTENSITY}" \
        ".env.alpha = ${alpha}" ".env.sight = ${sight}" \
        ".env.start_time = ${TEST_START}" ".env.max_step = ${TEST_TIME}" \
        ".agent.checkpoint_path = \"${test_ckpt}\"" \
        ".env.data_path = \"${type}.csv\"" \
        ".env.scaler_path = \"${model_path}/scaler.pkl\"" \
        ".env.gnn_path = \"${model_path}/best_model.pt\"" \
        ".seed = ${seed}"

    python -m code.train.simulation --config "$cfg"
}

# --- CB Adaptive: Train + Test ---

run_cb_adaptive_train_unit() {
    local type="$1" alpha="$2" sight="$3"
    local model_path=$(make_model_path "$type")
    local save_name=$(cb_adapt_save_name_train "$type" "$alpha" "$sight")
    local cfg="$CFG_ROOT/${save_name//\//_}.yaml"

    prepare_cfg "config/train_cb.yaml" "$cfg" \
        ".mode = \"train\"" ".train_mode = \"cb\"" \
        ".save_name = \"${save_name}\"" \
        ".env.attack_mode = \"adaptive\"" ".env.fixed_intensity = ${DEFAULT_FIXED_INTENSITY}" \
        ".env.alpha = ${alpha}" ".env.sight = ${sight}" \
        ".env.start_time = ${TRAIN_START}" ".env.max_step = ${TRAIN_TIME}" \
        ".train_episodes = ${TRAIN_EPISODES}" ".episodes = ${TRAIN_EPISODES}" \
        ".agent.gamma = 1.0" \
        ".env.data_path = \"${type}.csv\"" \
        ".env.scaler_path = \"${model_path}/scaler.pkl\"" \
        ".env.gnn_path = \"${model_path}/best_model.pt\""

    python -m code.train.simulation --config "$cfg"
}

run_cb_adaptive_test_unit() {
    local type="$1" alpha="$2" sight="$3" seed="$4"
    local model_path=$(make_model_path "$type")
    
    local train_dir="outputs/simulation/$(cb_adapt_save_name_train "$type" "$alpha" "$sight")"
    local test_ckpt="${train_dir}/final_agent.pt"
    
    local save_name=$(cb_adapt_save_name_test "$type" "$alpha" "$sight" "$seed")
    local cfg="$CFG_ROOT/${save_name//\//_}.yaml"

    prepare_cfg "config/test_cb.yaml" "$cfg" \
        ".mode = \"test\"" ".test_mode = \"cb\"" \
        ".save_name = \"${save_name}\"" \
        ".env.attack_mode = \"adaptive\"" ".env.fixed_intensity = ${DEFAULT_FIXED_INTENSITY}" \
        ".env.alpha = ${alpha}" ".env.sight = ${sight}" \
        ".env.start_time = ${TEST_START}" ".env.max_step = ${TEST_TIME}" \
        ".agent.checkpoint_path = \"${test_ckpt}\"" \
        ".env.data_path = \"${type}.csv\"" \
        ".env.scaler_path = \"${model_path}/scaler.pkl\"" \
        ".env.gnn_path = \"${model_path}/best_model.pt\"" \
        ".seed = ${seed}"

    python -m code.train.simulation --config "$cfg"
}

# =========================================================
# 主流程 (Train → Test)
# =========================================================
main() {
    echo "🚀 开始 Train+Test 全流程实验 (根目录: outputs/simulation, SEED_REPS=${SEED_REPS})"
    echo "📦 路径规范: a=攻击强度 (fixed), al=alpha (adaptive) | train 剔除 ts/t | test 剔除 tr/tt/ep"

    # =========================================================
    # 阶段 1: Threshold 测试 (无训练)
    # =========================================================
    echo -e "\n================ 阶段 1: Threshold 固定攻击测试 ================"
    for type in "${TH_TYPES[@]}"; do
        for atk in "${TH_FIXED_ATTACKS[@]}"; do
            for ((seed=1; seed<=SEED_REPS; seed++)); do
                echo "[🌱 Seed $seed/$SEED_REPS] th_fixed_${type}_a$(norm_num "$atk")" >&2
                launch_job "th_fixed_${type}_a$(norm_num "$atk")_s${seed}" \
                    run_th_fixed_unit "$type" "$atk" "$seed"
            done
        done
    done
    wait_all_jobs

    echo -e "\n================ 阶段 2: Threshold 自适应阈值测试 ================"
    for type in "${TH_TYPES[@]}"; do
        for thr in "${TH_ADAPTIVE_THRESHOLDS[@]}"; do
            for ((seed=1; seed<=SEED_REPS; seed++)); do
                echo "[🌱 Seed $seed/$SEED_REPS] th_adapt_${type}_thr$(norm_num "$thr")" >&2
                launch_job "th_adapt_${type}_thr$(norm_num "$thr")_s${seed}" \
                    run_th_adaptive_unit "$type" "$thr" "$seed"
            done
        done
    done
    wait_all_jobs

    # =========================================================
    # 阶段 3: RL Fixed 训练 → 测试
    # =========================================================
    echo -e "\n================ 阶段 3: RL Fixed 训练 ================"
    for type in "${RL_FIXED_TYPES[@]}"; do
        for atk in "${RL_FIXED_ATTACKS[@]}"; do
            launch_job "rl_fixed_train_${type}_a$(norm_num "$atk")" \
                run_rl_fixed_train_unit "$type" "$atk"
        done
    done
    wait_all_jobs

    echo -e "\n================ 阶段 4: RL Fixed 测试 (10 seeds) ================"
    for type in "${RL_FIXED_TYPES[@]}"; do
        for atk in "${RL_FIXED_ATTACKS[@]}"; do
            for ((seed=1; seed<=SEED_REPS; seed++)); do
                echo "[🌱 Seed $seed/$SEED_REPS] rl_fixed_${type}_a$(norm_num "$atk")" >&2
                launch_job "rl_fixed_${type}_a$(norm_num "$atk")_s${seed}" \
                    run_rl_fixed_test_unit "$type" "$atk" "$seed"
            done
        done
    done
    wait_all_jobs

    # =========================================================
    # 阶段 5: RL Adaptive 训练 → 测试
    # =========================================================
    echo -e "\n================ 阶段 5: RL Adaptive 训练 ================"
    for type in "${RL_ADAPTIVE_TYPES[@]}"; do
        for alpha in "${RL_ADAPTIVE_ALPHAS[@]}"; do
            for sight in "${RL_ADAPTIVE_SIGHTS[@]}"; do
                launch_job "rl_adapt_train_${type}_al$(norm_num "$alpha")_s$(norm_num "$sight")" \
                    run_rl_adaptive_train_unit "$type" "$alpha" "$sight"
            done
        done
    done
    wait_all_jobs

    echo -e "\n================ 阶段 6: RL Adaptive 测试 (10 seeds) ================"
    for type in "${RL_ADAPTIVE_TYPES[@]}"; do
        for alpha in "${RL_ADAPTIVE_ALPHAS[@]}"; do
            for sight in "${RL_ADAPTIVE_SIGHTS[@]}"; do
                for ((seed=1; seed<=SEED_REPS; seed++)); do
                    echo "[🌱 Seed $seed/$SEED_REPS] rl_adapt_${type}_al$(norm_num "$alpha")_s$(norm_num "$sight")" >&2
                    launch_job "rl_adapt_${type}_al$(norm_num "$alpha")_s$(norm_num "$sight")_s${seed}" \
                        run_rl_adaptive_test_unit "$type" "$alpha" "$sight" "$seed"
                done
            done
        done
    done
    wait_all_jobs

    # =========================================================
    # 阶段 7: CB Adaptive 训练 → 测试
    # =========================================================
    echo -e "\n================ 阶段 7: CB Adaptive 训练 ================"
    for type in "${CB_ADAPTIVE_TYPES[@]}"; do
        for alpha in "${CB_ADAPTIVE_ALPHAS[@]}"; do
            for sight in "${CB_ADAPTIVE_SIGHTS[@]}"; do
                launch_job "cb_adapt_train_${type}_al$(norm_num "$alpha")_s$(norm_num "$sight")" \
                    run_cb_adaptive_train_unit "$type" "$alpha" "$sight"
            done
        done
    done
    wait_all_jobs

    echo -e "\n================ 阶段 8: CB Adaptive 测试 (10 seeds) ================"
    for type in "${CB_ADAPTIVE_TYPES[@]}"; do
        for alpha in "${CB_ADAPTIVE_ALPHAS[@]}"; do
            for sight in "${CB_ADAPTIVE_SIGHTS[@]}"; do
                for ((seed=1; seed<=SEED_REPS; seed++)); do
                    echo "[🌱 Seed $seed/$SEED_REPS] cb_adapt_${type}_al$(norm_num "$alpha")_s$(norm_num "$sight")" >&2
                    launch_job "cb_adapt_${type}_al$(norm_num "$alpha")_s$(norm_num "$sight")_s${seed}" \
                        run_cb_adaptive_test_unit "$type" "$alpha" "$sight" "$seed"
                done
            done
        done
    done
    wait_all_jobs

    # =========================================================
    # 总结
    # =========================================================
    echo -e "\n================ 实验总结 ================"
    echo "✅ 成功: $(wc -l < "$SUCCESS_LOG") | ❌ 失败: $(wc -l < "$FAILED_LOG")"
    [[ -s "$FAILED_LOG" ]] && echo -e "\n❌ 失败列表:\n$(cat "$FAILED_LOG")"
    echo -e "\n📁 日志目录: $LOG_ROOT"
    echo -e "📂 输出目录: outputs/simulation/{train,test}/"
}

main