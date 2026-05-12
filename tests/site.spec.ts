import { expect, test } from '@playwright/test';

const MAX_NAV_LINKS = 10;
const SITE_ORIGIN = 'http://localhost:8000';

function localNavigablePath(href: string | null): string | null {
  if (!href) return null;
  if (href.startsWith('#')) return null;
  if (href.startsWith('mailto:')) return null;
  if (href.startsWith('tel:')) return null;
  if (href.startsWith('javascript:')) return null;

  const url = new URL(href, SITE_ORIGIN);
  if (url.origin !== SITE_ORIGIN) return null;
  if (url.hash && (url.pathname === '/' || url.pathname === '/index.html')) return null;
  return `${url.pathname}${url.search}${url.hash}`;
}

test('homepage loads', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveTitle(/./);
  await expect(page.locator('body')).toBeVisible();
});

test('navigation works', async ({ page }) => {
  await page.goto('/');
  const links = page.locator('nav a[href], a.md-nav__link[href], main a[href]');
  const count = await links.count();
  const hrefs: string[] = [];

  for (let i = 0; i < count; i += 1) {
    const href = await links.nth(i).getAttribute('href');
    const localPath = localNavigablePath(href);
    if (localPath) hrefs.push(localPath);
  }

  const seen = new Set<string>();
  let checked = 0;

  for (const href of hrefs) {
    if (checked >= MAX_NAV_LINKS) break;
    if (seen.has(href)) continue;

    seen.add(href);
    checked += 1;
    await page.goto(href);
    await expect(page.locator('body')).toBeVisible();
    await expect(page.locator('body')).not.toContainText(/404\s*-\s*Not found|Page not found/i);
  }

  if (checked === 0) {
    throw new Error('No local navigation links were found to check.');
  }
});

test('no obvious 404 text on key pages', async ({ page }) => {
  const paths = [
    '/',
    '/time_series/prism_tipping_point_forecast/',
    '/remote_sensing/post_fire_tipping_points_random_forest/',
    '/how-to-contribute/'
  ];

  for (const path of paths) {
    await page.goto(path);
    await expect(page.locator('body')).toBeVisible();
    await expect(page.locator('body')).not.toContainText(/404\s*-\s*Not found|Page not found/i);
  }
});

test('core content is usable', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: /analytics library/i })).toBeVisible();

  const mainLinks = await page.locator('main a').count();
  if (mainLinks < 3) {
    console.warn(`Homepage has only ${mainLinks} main-content links; consider adding clearer entry points.`);
  }
});
