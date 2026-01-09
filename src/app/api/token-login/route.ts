import { NextResponse } from 'next/server';
import {
  TOKEN_COOKIE_NAME,
  getAllowedTokens,
  getTokenCookieOptions,
} from '@/lib/token-auth';

export async function POST(request: Request) {
  const allowedTokens = getAllowedTokens();
  if (allowedTokens.length === 0) {
    return NextResponse.json(
      { ok: false, error: 'Token login not configured.' },
      { status: 500 }
    );
  }

  const body = await request.json().catch(() => ({}));
  const token = typeof body.token === 'string' ? body.token.trim() : '';

  if (!token || !allowedTokens.includes(token)) {
    return NextResponse.json({ ok: false }, { status: 401 });
  }

  const response = NextResponse.json({ ok: true });
  response.cookies.set(TOKEN_COOKIE_NAME, token, getTokenCookieOptions());
  return response;
}
