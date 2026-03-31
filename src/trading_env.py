from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from dataset_builder import BASIC_SENTIMENT_COLUMNS, ENHANCED_SENTIMENT_COLUMNS

TARGET_ALLOC = 0.5

MARKET_FEATURES = [
    'return_1d',
    'return_3d',
    'rolling_volatility_5d',
]

SENTIMENT_BASIC_FEATURES = BASIC_SENTIMENT_COLUMNS
SENTIMENT_ENHANCED_FEATURES = ENHANCED_SENTIMENT_COLUMNS

PRICE_FEATURES = [
    'open', 'high', 'low', 'close', 'volume',
]


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        use_sentiment: bool = False,
        sentiment_feature_set: str | None = None,
        initial_cash: float = 10_000.0,
        transaction_cost: float = 0.001,
        target_alloc: float = TARGET_ALLOC,
        reward_mode: str = "log_return",
        drawdown_penalty: float = 0.0,
        volatility_penalty: float = 0.0,
        volatility_window: int = 5,
    ):
        super().__init__()

        self.data = data.reset_index(drop=True).copy()
        self.sentiment_feature_set = self._resolve_sentiment_feature_set(
            use_sentiment=use_sentiment,
            sentiment_feature_set=sentiment_feature_set,
        )
        self.use_sentiment = self.sentiment_feature_set is not None
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.target_alloc = target_alloc
        self.reward_mode = reward_mode
        self.drawdown_penalty = drawdown_penalty
        self.volatility_penalty = volatility_penalty
        self.volatility_window = volatility_window

        self.feature_cols = MARKET_FEATURES + PRICE_FEATURES
        if self.sentiment_feature_set == "basic":
            self.feature_cols = self.feature_cols + SENTIMENT_BASIC_FEATURES
        elif self.sentiment_feature_set == "enhanced":
            self.feature_cols = self.feature_cols + SENTIMENT_ENHANCED_FEATURES

        for col in self.feature_cols:
            if col not in self.data.columns:
                self.data[col] = 0.0

        self._raw = self.data[self.feature_cols].copy().fillna(0)
        self._norm = self._normalize(self._raw)

        obs_dim = len(self.feature_cols) + 2

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._step = 0
        self._position = 0
        self._cash = initial_cash
        self._portfolio_value = initial_cash
        self._peak_portfolio_value = initial_cash
        self._history = []
        self._returns = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        self._position = 0
        self._cash = self.initial_cash
        self._portfolio_value = self.initial_cash
        self._peak_portfolio_value = self.initial_cash
        self._history = []
        self._returns = []
        return self._get_obs(), {}

    def step(self, action):
        assert 0 <= action <= 2, f"Invalid action {action}"

        price = self._current_price()
        prev_value = self._portfolio_value
        cost = 0.0

        if action == 0:
            if self._position > 0:
                proceeds = self._position * price * (1 - self.transaction_cost)
                cost = self._position * price * self.transaction_cost
                self._cash += proceeds
                self._position = 0

        elif action == 2:
            target_value = self._portfolio_value * self.target_alloc
            current_stock_value = self._position * price
            gap = target_value - current_stock_value
            if gap > 0 and self._cash > price * (1 + self.transaction_cost):
                affordable = self._cash / (price * (1 + self.transaction_cost))
                needed = gap / price
                shares_to_buy = int(min(affordable, needed))
                if shares_to_buy > 0:
                    total_cost = shares_to_buy * price * (1 + self.transaction_cost)
                    cost = shares_to_buy * price * self.transaction_cost
                    self._cash -= total_cost
                    self._position += shares_to_buy

        self._step += 1
        done = self._step >= len(self.data) - 1

        new_price = self._current_price()
        self._portfolio_value = self._cash + self._position * new_price

        daily_return = float(self._portfolio_value / (prev_value + 1e-8) - 1.0)
        self._returns.append(daily_return)
        reward = self._compute_reward(prev_value, daily_return)
        if cost > 0:
            reward -= cost / prev_value

        self._peak_portfolio_value = max(self._peak_portfolio_value, self._portfolio_value)
        if self.drawdown_penalty > 0:
            drawdown = max(
                0.0,
                (self._peak_portfolio_value - self._portfolio_value)
                / (self._peak_portfolio_value + 1e-8),
            )
            reward -= self.drawdown_penalty * drawdown

        if self.volatility_penalty > 0 and len(self._returns) >= 2:
            recent = self._returns[-self.volatility_window :]
            reward -= self.volatility_penalty * float(np.std(recent))

        self._history.append({
            'step':            self._step,
            'price':           new_price,
            'position':        self._position,
            'cash':            self._cash,
            'portfolio_value': self._portfolio_value,
            'reward':          reward,
            'daily_return':    daily_return,
            'action':          action,
        })

        obs = self._get_obs()
        info = {'portfolio_value': self._portfolio_value, 'position': self._position}
        return obs, reward, done, False, info

    def render(self):
        pass

    def _get_obs(self):
        step = min(self._step, len(self._norm) - 1)
        features = self._norm.iloc[step].values.astype(np.float32)
        stock_value = self._position * self._current_price()
        invested_frac = stock_value / (self._portfolio_value + 1e-8)
        portfolio_gain = self._portfolio_value / self.initial_cash - 1.0
        return np.concatenate([features, [invested_frac, portfolio_gain]]).astype(np.float32)

    def _current_price(self):
        step = min(self._step, len(self.data) - 1)
        return float(self.data.iloc[step]['close'])

    def _compute_reward(self, prev_value: float, daily_return: float) -> float:
        if self.reward_mode == "simple_return":
            return daily_return
        if self.reward_mode == "log_return":
            gross_return = max(self._portfolio_value / (prev_value + 1e-8), 1e-8)
            return float(np.log(gross_return))
        raise ValueError(f"Unsupported reward_mode: {self.reward_mode}")

    @staticmethod
    def _resolve_sentiment_feature_set(
        use_sentiment: bool,
        sentiment_feature_set: str | None,
    ) -> str | None:
        if sentiment_feature_set is not None:
            if sentiment_feature_set not in {"basic", "enhanced"}:
                raise ValueError(
                    "sentiment_feature_set must be None, 'basic', or 'enhanced'"
                )
            return sentiment_feature_set
        return "basic" if use_sentiment else None

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for col in df.columns:
            mu = df[col].mean()
            sigma = df[col].std()
            if sigma > 1e-8:
                result[col] = ((df[col] - mu) / sigma).clip(-5, 5)
            else:
                result[col] = 0.0
        return result

    def get_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._history)


def make_envs(
    daily_df: pd.DataFrame,
    ticker: str,
    train_cutoff: str = '2025-11-30',
    env_kwargs: dict | None = None,
):
    t = daily_df[daily_df['ticker'] == ticker].copy()
    train = t[t['date'] <= train_cutoff]
    test  = t[t['date'] >  train_cutoff]
    env_kwargs = env_kwargs or {}

    return {
        'train_price':               TradingEnv(train, use_sentiment=False, **env_kwargs),
        'train_sentiment':           TradingEnv(train, sentiment_feature_set="basic", **env_kwargs),
        'test_price':                TradingEnv(test,  use_sentiment=False, **env_kwargs),
        'test_sentiment':            TradingEnv(test,  sentiment_feature_set="basic", **env_kwargs),
        'train_sentiment_basic':     TradingEnv(train, sentiment_feature_set="basic", **env_kwargs),
        'test_sentiment_basic':      TradingEnv(test,  sentiment_feature_set="basic", **env_kwargs),
        'train_sentiment_enhanced':  TradingEnv(train, sentiment_feature_set="enhanced", **env_kwargs),
        'test_sentiment_enhanced':   TradingEnv(test,  sentiment_feature_set="enhanced", **env_kwargs),
    }
