export const TOKEN_COOKIE_NAME = 'tokenAuth';
export const TOKEN_COOKIE_MAX_AGE = 60 * 60 * 8;

export function getAllowedTokens() {
  const rawTokens = process.env.ACCESS_TOKENS ?? process.env.ACCESS_TOKEN ?? '';
  return rawTokens
    .split(',')
    .map((token) => token.trim())
    .filter(Boolean);
}

export function getTokenCookieOptions(maxAge = TOKEN_COOKIE_MAX_AGE) {
  return {
    httpOnly: true,
    sameSite: 'lax' as const,
    path: '/',
    secure: process.env.NODE_ENV === 'production',
    maxAge,
  };
}
