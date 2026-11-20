import { tmpdir } from 'node:os';
import { join } from 'node:path';

/** Where test audio and reference decodes go (override with $PICOTONE_TEST_DIR or --dir). */
export const DEFAULT_TEST_DIR = process.env.PICOTONE_TEST_DIR ?? join(tmpdir(), 'picotone-web-test');
