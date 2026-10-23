/** Register the product's exact campus tool roster in a pinned Harness profile.
 *
 * The parent experiment runner writes the existing OpenAI tool schemas to a
 * local file and serves the existing Python tool executor on loopback. The
 * plugin cannot access tools absent from that file or substitute canned data.
 */
import { readFileSync } from 'node:fs'

export const name = 'campus-experiment-tools'
export const inject = ['tools']

export function apply(ctx) {
  const rawLimit = process.env.CAMPUS_MAX_MODEL_REQUESTS_PER_TURN
  if (rawLimit !== undefined) {
    const limit = Number(rawLimit)
    if (!Number.isInteger(limit) || limit < 1 || limit > 20) {
      throw new Error('invalid campus model request limit')
    }
    const calls = new Map()
    ctx.on('agent/pre-step', ({ agent, step }, next) => {
      if (step === 1) calls.set(String(agent.id), 0)
      return next()
    })
    ctx.on('llm/stream', (request, next) => {
      if (request.sessionId === undefined || request.purpose) return next()
      const session = String(request.sessionId)
      const used = calls.get(session) ?? 0
      if (used >= limit) throw new Error('campus_model_request_budget_exceeded')
      calls.set(session, used + 1)
      return next()
    })
  }
  const path = process.env.CAMPUS_TOOL_SCHEMA_PATH
  const endpoint = process.env.CAMPUS_TOOL_BRIDGE_URL
  const token = process.env.CAMPUS_TOOL_BRIDGE_TOKEN
  if (!path || !endpoint?.startsWith('http://127.0.0.1:') || !token) {
    throw new Error('campus experiment tool bridge is not configured')
  }
  const rows = JSON.parse(readFileSync(path, 'utf8'))
  if (!Array.isArray(rows) || rows.length === 0) {
    throw new Error('campus tool schemas are empty')
  }
  const seen = new Set()
  for (const row of rows) {
    const spec = row?.function
    if (row?.type !== 'function' || typeof spec?.name !== 'string' ||
        typeof spec?.description !== 'string' || !spec?.parameters ||
        seen.has(spec.name)) {
      throw new Error('invalid or duplicate campus tool schema')
    }
    seen.add(spec.name)
    ctx.tools.register({
      name: spec.name,
      description: spec.description,
      parameters: spec.parameters,
      output: {
        schema: { type: 'string' },
        render(_args, value) { return [{ type: 'text', text: value }] },
      },
      async execute(args) {
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-Campus-Token': token },
          body: JSON.stringify({ name: spec.name, arguments: args }),
          signal: AbortSignal.timeout(45000),
        })
        if (!response.ok) throw new Error(`campus tool bridge HTTP ${response.status}`)
        const body = await response.json()
        if (typeof body?.result !== 'string') throw new Error('invalid campus tool response')
        return body.result
      },
    })
  }
}
