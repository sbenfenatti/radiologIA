import { NextRequest, NextResponse } from 'next/server';
import {
  TOKEN_COOKIE_NAME,
  getAllowedTokens,
  getTokenCookieOptions,
} from '@/lib/token-auth';

export async function GET(request: NextRequest) {
  const allowedTokens = getAllowedTokens();
  if (allowedTokens.length === 0) {
    return NextResponse.json(
      { ok: false, error: 'Token login not configured.' },
      { status: 500 }
    );
  }

  const token = request.cookies.get(TOKEN_COOKIE_NAME)?.value ?? '';
  if (!token || !allowedTokens.includes(token)) {
    return NextResponse.json({ ok: false }, { status: 401 });
  }

  return NextResponse.json({ ok: true });
}

export async function DELETE() {
  const response = NextResponse.json({ ok: true });
  response.cookies.set(TOKEN_COOKIE_NAME, '', getTokenCookieOptions(0));
  return response;
}
