import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  base: './',
  plugins: [react()],
  server: {
    allowedHosts: ['launcher-t6oubm23hi-codeserver.ide.taiji.woa.com'],
  },
  preview: {
    allowedHosts: ['launcher-t6oubm23hi-codeserver.ide.taiji.woa.com'],
  },
})
