const ACTIVE_GAME_STORAGE_KEY = "chess-llm-workbench.active-game-id";

export function readActiveGameId(storage: Storage = window.localStorage): string | null {
  const value = storage.getItem(ACTIVE_GAME_STORAGE_KEY)?.trim();
  return value ? value : null;
}

export function writeActiveGameId(gameId: string, storage: Storage = window.localStorage): void {
  storage.setItem(ACTIVE_GAME_STORAGE_KEY, gameId);
}

export function clearActiveGameId(storage: Storage = window.localStorage): void {
  storage.removeItem(ACTIVE_GAME_STORAGE_KEY);
}
