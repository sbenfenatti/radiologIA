'use client';

import { useEffect, useRef } from 'react';

const INACTIVITY_MS = 60 * 60 * 1000;
const ABSOLUTE_MS = 8 * 60 * 60 * 1000;
const AUTH_START_KEY = 'radiologia.auth.start';
const AUTH_LAST_KEY = 'radiologia.auth.last';
const AUTH_ID_KEY = 'radiologia.auth.id';
const ACTIVITY_THROTTLE_MS = 15000;
const CHECK_INTERVAL_MS = 60000;

type SessionExpiryOptions = {
  isActive: boolean;
  identity?: string | null;
  onExpire: () => void;
};

export const useSessionExpiry = ({ isActive, identity, onExpire }: SessionExpiryOptions) => {
  const expireCalledRef = useRef(false);
  const lastActivityRef = useRef(0);

  useEffect(() => {
    if (!isActive) {
      expireCalledRef.current = false;
      return;
    }

    const now = Date.now();
    const identityValue = identity ?? 'unknown';
    const storedId = localStorage.getItem(AUTH_ID_KEY);
    if (storedId !== identityValue) {
      localStorage.setItem(AUTH_ID_KEY, identityValue);
      localStorage.setItem(AUTH_START_KEY, String(now));
      localStorage.setItem(AUTH_LAST_KEY, String(now));
    } else {
      if (!localStorage.getItem(AUTH_START_KEY)) {
        localStorage.setItem(AUTH_START_KEY, String(now));
      }
      if (!localStorage.getItem(AUTH_LAST_KEY)) {
        localStorage.setItem(AUTH_LAST_KEY, String(now));
      }
    }
    lastActivityRef.current = now;

    const expire = () => {
      if (expireCalledRef.current) {
        return;
      }
      expireCalledRef.current = true;
      localStorage.removeItem(AUTH_START_KEY);
      localStorage.removeItem(AUTH_LAST_KEY);
      localStorage.removeItem(AUTH_ID_KEY);
      onExpire();
    };

    const checkExpiry = () => {
      if (expireCalledRef.current) {
        return;
      }
      const now = Date.now();
      const start = Number(localStorage.getItem(AUTH_START_KEY) ?? '0');
      const last = Number(localStorage.getItem(AUTH_LAST_KEY) ?? '0');
      if (start && now - start >= ABSOLUTE_MS) {
        expire();
        return;
      }
      if (last && now - last >= INACTIVITY_MS) {
        expire();
      }
    };

    const updateActivity = () => {
      if (!isActive) {
        return;
      }
      const now = Date.now();
      if (now - lastActivityRef.current < ACTIVITY_THROTTLE_MS) {
        return;
      }
      lastActivityRef.current = now;
      localStorage.setItem(AUTH_LAST_KEY, String(now));
    };

    checkExpiry();
    const interval = window.setInterval(checkExpiry, CHECK_INTERVAL_MS);
    const events: (keyof WindowEventMap)[] = [
      'click',
      'keydown',
      'mousemove',
      'scroll',
      'touchstart',
    ];
    events.forEach((event) => window.addEventListener(event, updateActivity, { passive: true }));
    window.addEventListener('visibilitychange', updateActivity);

    const handleStorage = (event: StorageEvent) => {
      if (event.key === AUTH_START_KEY || event.key === AUTH_LAST_KEY) {
        checkExpiry();
      }
    };
    window.addEventListener('storage', handleStorage);

    return () => {
      window.clearInterval(interval);
      events.forEach((event) => window.removeEventListener(event, updateActivity));
      window.removeEventListener('visibilitychange', updateActivity);
      window.removeEventListener('storage', handleStorage);
    };
  }, [identity, isActive, onExpire]);
};
