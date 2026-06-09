import { TASK_TARGET } from '../core/constants.js';

const REWARD = {
  scoreScale: 0.01,
  chainScale: 0.005,
  taskDelta: 2.0,
  emptySwap: -0.05,
  stepCost: -0.02,
  winBonus: 50.0,
  stepsLeftBonus: 0.2,
  losePenalty: -10.0,
  taskDeficitPenalty: -3.0,
};

export function computeReward(prevState, result, nextState) {
  let r = 0;

  r += (result.totalScore || 0) * REWARD.scoreScale;
  r += (result.chainScore || 0) * REWARD.chainScale;

  for (const shape of nextState.targetShapes) {
    const prev = prevState.taskScores[shape] || 0;
    const next = nextState.taskScores[shape] || 0;
    r += (next - prev) * REWARD.taskDelta;
  }

  const usedPowerup = Boolean(result.usedPowerup);
  if (!result.hadMatch && !usedPowerup) {
    r += REWARD.emptySwap;
  }

  r += REWARD.stepCost;

  if (nextState.won) {
    r += REWARD.winBonus;
    r += Math.max(0, nextState.totalSteps - nextState.stepsUsed) * REWARD.stepsLeftBonus;
  } else if (nextState.over) {
    r += REWARD.losePenalty;
    let deficit = 0;
    for (const shape of nextState.targetShapes) {
      deficit += Math.max(0, TASK_TARGET - (nextState.taskScores[shape] || 0));
    }
    r += deficit * REWARD.taskDeficitPenalty;
  }

  return r;
}

export { REWARD };
