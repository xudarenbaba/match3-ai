/**
 * 调用本地 Python RL 推理服务获取走棋
 */

import { buildObservation, FRAME_STACK } from '../rl/observation.js';

export const RL_API = 'http://127.0.0.1:8765';

// 帧历史缓冲，存储最近 FRAME_STACK 帧的单帧 obs（board+global 数组）
// 新局开始时调用 resetFrameHistory() 清空
let _frameHistory = [];

/** 新局开始时重置帧历史 */
export function resetFrameHistory() {
  _frameHistory = [];
}

export function serializeState(state) {
  return {
    board: state.board,
    targetShapes: state.targetShapes,
    taskScores: state.taskScores,
    score: state.score,
    chainScoreTotal: state.chainScoreTotal,
    totalSteps: state.totalSteps,
    stepsUsed: state.stepsUsed,
    lastAction: state.lastAction ?? -1,
    taskTarget: state.taskTarget ?? 4,
    won: state.won,
    over: state.over,
  };
}

export async function checkRlServer() {
  try {
    const res = await fetch(`${RL_API}/health`, { signal: AbortSignal.timeout(2000) });
    if (!res.ok) return false;
    const data = await res.json();
    return Boolean(data.ok && data.model_loaded);
  } catch {
    return false;
  }
}

export async function findRlMove(state) {
  // 构建当前帧并压入历史
  const currentFrame = buildObservation(state);
  _frameHistory.push(currentFrame);
  if (_frameHistory.length > FRAME_STACK) {
    _frameHistory = _frameHistory.slice(-FRAME_STACK);
  }

  // 把历史帧和当前局面状态一起发给服务端
  const payload = {
    ...serializeState(state),
    frameHistory: _frameHistory,  // 服务端用这个做帧堆叠
  };

  const res = await fetch(`${RL_API}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const err = await res.json();
      if (err.error) msg = err.error;
    } catch {
      /* ignore */
    }
    throw new Error(msg);
  }
  const data = await res.json();
  return {
    from: data.from,
    to: data.to,
    reason: data.reason || 'RL 策略',
  };
}
