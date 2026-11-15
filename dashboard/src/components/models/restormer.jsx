import { useState } from 'react'
import { Figure, Swatch, fmtDelta, useInView } from '../ui.jsx'
import { GroupedBars, Trend } from '../charts.jsx'
import { ModelDiagram } from '../diagrams.jsx'
import { LevelFigures, Setup, Takeaways } from '../modelkit.jsx'
import { HEX, img, restormerFt } from '../../data/project.js'
import { restormerLevels as L } from '../../data/restormerLevels.js'

// Phase 2: one fine-tuned Restormer per HDC level, Levels 0–9, 40 test pages each.
// Every number comes from "Restormer Fine Tuned/report/tables/" via restormerLevels.js.

const METHODS = [
  { key: 'input', label: 'Blurred input', color: HEX.input },
  { key: 'pre', label: 'Pretrained Restormer', color: HEX.pre },
  { key: 'ft', label: 'Fine-tuned Restormer', color: HEX.ft },
  { key: 'gt', label: 'Sharp ground truth', color: HEX.gt },
]
const OCR_LEVELS = L.filter((r) => r.ft.char != null)
const pad = (n) => String(n).padStart(2, '0')

const imageSeries = (metric) => L.map((r) => ({
  level: `L${r.level}`, input: r.input[metric], pre: r.pre[metric], ft: r.ft[metric],
}))
const ocrSeries = (metric) => OCR_LEVELS.map((r) => ({
  level: `L${r.level}`, input: r.input[metric], pre: r.pre[metric], ft: r.ft[metric], gt: r.gt[metric],
}))
const fontSeries = OCR_LEVELS.map((r) => ({
  level: `L${r.level}`,
  times: r.byFont.ft.timesR.char, verdana: r.byFont.ft.verdanaRef.char,
  gtTimes: r.byFont.gt.timesR.char, gtVerdana: r.byFont.gt.verdanaRef.char,
}))

// One short reading per level, condensed from report/PROJECT_CONTEXT.md §6.
const NOTES = {
  0: ['The mildest step: the blurred input already scores 26.08 dB against the sharp capture.', 'Fine-tuning still adds +2.66 dB and +0.167 SSIM, and every test image improves.', 'OCR was not saved for this run, so Level 0 has image metrics only.'],
  1: ['EasyOCR already reads the blurred page at 98.4%, so there is nothing to recover.', 'Restoration lowers OCR to 92.6%: 28 of 40 pages get worse. The restored page looks like the sharp capture, and EasyOCR drops wide glyphs on those (the sharp image itself reads 90.8%).', 'Image quality still rises sharply: +5.70 dB, +0.207 SSIM.'],
  2: ['The first level where restoration helps OCR: 94.2% → 96.1% of characters and 63.5% → 75.0% of tokens.', 'PSNR jumps by +8.01 dB.', 'The blur estimate for Levels 0–2 sits at the 5 px floor of the search grid, so σ here is a lower bound.'],
  3: ['The best OCR result of the study: 98.3% of characters, 86.5% of tokens, 62.5% of pages read with no errors.', 'The largest PSNR gain, +8.38 dB.', 'The restored page reads better than the sharp capture (92.5%).'],
  4: ['The blurred page is unreadable in practice: 23.5% of characters, 0.5% of tokens.', 'Restoration recovers it almost completely: 97.4% of characters, and all 40 pages improve.', 'Registration starts to matter: the alignment score drops to 0.915.'],
  5: ['EasyOCR finds no text at all on the blurred input or the pretrained output.', 'After fine-tuning it reads 96.9% of characters and 75.0% of tokens; 40% of pages are perfect.', 'The first run, with an earlier notebook: its date, σ ≈ 25 px and 128 px × 4 patches are inferred.'],
  6: ['A clear drop from Level 5: 83.2% of characters and 32.0% of tokens; no page is read perfectly.', 'A likely cause: σ ≈ 30 px still gets 128 px patches, only about 4σ wide, smaller than the blur’s footprint.', 'Fonts diverge: Times 78.5% against Verdana 87.9%.'],
  7: ['Still usable at character level (77.0%), weak at token level (22.3%).', 'From here patches grow to 192 px with batch 2: 6,000 patches seen instead of 12,000, and about 81 minutes per run.', 'Times 66.3% against Verdana 87.7%.'],
  8: ['Similar image quality to Level 7 (19.47 dB, SSIM 0.563); 73.5% of characters.', 'Almost all of the loss is on the serif font: Times 57.5% against Verdana 89.5%.', 'The blurred input’s SSIM is higher than at Level 7, so compare SSIM within a level, not across levels.'],
  9: ['Near the method’s limit: +2.42 dB, 52.8% of characters, 10.9% of tokens.', 'Training was unstable and under-trained: the best checkpoint is at iteration 2,000 and the loss was still falling at 3,000.', 'Alignment is the weakest (0.780), and Times outputs show invented strokes (34.3% against Verdana 71.3%).'],
}

/* ---------------- Level explorer ---------------- */

function OcrBars({ row }) {
  const [ref, inView] = useInView()
  return (
    <div className="bars" ref={ref}>
      {METHODS.map((m) => {
        const v = row[m.key].char
        return (
          <div className="bar-row" key={m.key}>
            <span className="method-cell"><Swatch color={m.color} />{m.label}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: inView ? `${v}%` : 0, background: m.color }} />
            </div>
            <output>{v.toFixed(1)}%</output>
          </div>
        )
      })}
    </div>
  )
}

function LevelExplorer() {
  const [lv, setLv] = useState(5)
  const [view, setView] = useState('comparison')
  const r = L[lv]
  const hasOcr = r.ft.char != null
  return (
    <>
      <div className="cmp-toolbar">
        <div className="seg seg-levels" role="tablist" aria-label="Blur level">
          {L.map((x) => (
            <button key={x.level} role="tab" aria-selected={x.level === lv} onClick={() => setLv(x.level)}>L{x.level}</button>
          ))}
        </div>
        <span className="small muted">σ ≈ {r.train.sigma} px{r.train.inferred ? ' (inferred)' : ''} · 40 test pages</span>
      </div>

      <div className="sample-stats level-stats">
        <div><b>{r.input.psnr.toFixed(2)} → {r.ft.psnr.toFixed(2)}</b><span>PSNR, dB ({fmtDelta(r.ft.psnr - r.input.psnr)})</span></div>
        <div><b>{r.input.ssim.toFixed(3)} → {r.ft.ssim.toFixed(3)}</b><span>SSIM ({fmtDelta(r.ft.ssim - r.input.ssim, 3)})</span></div>
        <div><b className={hasOcr && r.ft.char >= r.input.char ? 'gain' : ''}>{hasOcr ? `${r.input.char.toFixed(1)} → ${r.ft.char.toFixed(1)}%` : 'Not run'}</b><span>OCR characters read</span></div>
        <div><b>{hasOcr ? `${r.input.token.toFixed(1)} → ${r.ft.token.toFixed(1)}%` : '—'}</b><span>OCR tokens read</span></div>
        <div><b>{hasOcr ? `${r.gt.char.toFixed(1)}%` : '—'}</b><span>OCR on the sharp page</span></div>
        <div><b>{r.wins.psnr}/40</b><span>{r.wins.char ? `pages improved · OCR ${r.wins.char.join(' / ')}` : 'pages improved in PSNR'}</span></div>
      </div>

      <div className="two" style={{ alignItems: 'start' }}>
        <div className="panel">
          <h4>What happens at Level {lv}</h4>
          <ul className="facts small">{NOTES[lv].map((t) => <li key={t}>{t}</li>)}</ul>
        </div>
        <div className="panel">
          <h4>Character accuracy by method, Level {lv}</h4>
          {hasOcr ? <OcrBars key={lv} row={r} /> : <p className="small muted">No OCR was saved for Level 0.</p>}
          {hasOcr && r.wins.char && (
            <p className="chart-note">Against the blurred input, fine-tuning wins on {r.wins.char[0]}, ties on {r.wins.char[1]} and loses on {r.wins.char[2]} of the 40 pages.</p>
          )}
        </div>
      </div>

      <div className="cmp-toolbar" style={{ marginTop: 24 }}>
        <div className="seg" role="tablist" aria-label="Image view">
          <button role="tab" aria-selected={view === 'comparison'} onClick={() => setView('comparison')}>Full pages</button>
          <button role="tab" aria-selected={view === 'closeup'} onClick={() => setView('closeup')}>Close-ups</button>
        </div>
        <span className="small muted">Click to enlarge</span>
      </div>
      <Figure
        key={`${lv}-${view}`}
        src={img(`restormer/level_${pad(lv)}_${view}.jpg`)}
        caption={view === 'comparison'
          ? `Level ${lv}: blurred input, fine-tuned Restormer and registered ground truth for 4 test pages (2 Times, 2 Verdana), with per-image PSNR, SSIM and the EasyOCR transcript.`
          : `Level ${lv}: native-resolution centre crops of the same pages.`}
      />
    </>
  )
}

/* ---------------- Results across levels ---------------- */

function AllLevels() {
  const psnr = imageSeries('psnr')
  const ssim = imageSeries('ssim')
  const imgSeries = METHODS.slice(0, 3).map((m) => ({ key: m.key, name: m.label, color: m.color, dashed: m.key !== 'ft' }))
  return (
    <>
      <div className="two">
        <div className="panel">
          <h4>PSNR by level, dB</h4>
          <Trend data={psnr} xKey="level" domain={[12, 30]} series={imgSeries} />
          <p className="chart-note">Fine-tuning adds +2.4 to +8.4 dB at every level. The pretrained model never moves more than 0.05 dB above the blurred input.</p>
        </div>
        <div className="panel">
          <h4>SSIM by level</h4>
          <Trend data={ssim} xKey="level" domain={[0.3, 0.72]} digits={3} series={imgSeries} />
          <p className="chart-note">SSIM rises by +0.11 to +0.23. The input’s SSIM is not monotonic across levels, so compare within a level.</p>
        </div>
      </div>

      <div className="table-scroll" style={{ marginTop: 24 }}>
        <table>
          <thead>
            <tr>
              <th>Level</th><th className="num">σ (px)</th>
              <th className="num">Input PSNR</th><th className="num">Pretrained</th><th className="num">Fine-tuned</th><th className="num">Δ</th>
              <th className="num">Input SSIM</th><th className="num">Fine-tuned</th><th className="num">Δ</th>
              <th className="num">OCR blurred</th><th className="num">OCR fine-tuned</th><th className="num">OCR sharp</th>
            </tr>
          </thead>
          <tbody>
            {L.map((r) => (
              <tr key={r.level} className={r.level === 5 ? 'hl' : ''}>
                <td>L{r.level}</td>
                <td className="num">{r.train.sigma}{r.train.inferred ? '*' : ''}</td>
                <td className="num">{r.input.psnr.toFixed(2)}</td>
                <td className="num">{r.pre.psnr.toFixed(2)}</td>
                <td className="num"><b>{r.ft.psnr.toFixed(2)}</b></td>
                <td className="num delta-pos">{fmtDelta(r.ft.psnr - r.input.psnr)}</td>
                <td className="num">{r.input.ssim.toFixed(3)}</td>
                <td className="num"><b>{r.ft.ssim.toFixed(3)}</b></td>
                <td className="num delta-pos">{fmtDelta(r.ft.ssim - r.input.ssim, 3)}</td>
                <td className="num">{r.input.char == null ? '—' : `${r.input.char.toFixed(1)}%`}</td>
                <td className={`num ${r.ft.char != null && r.ft.char < r.input.char ? 'delta-neg' : ''}`}><b>{r.ft.char == null ? '—' : `${r.ft.char.toFixed(1)}%`}</b></td>
                <td className="num">{r.gt?.char == null ? '—' : `${r.gt.char.toFixed(1)}%`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="chart-note">
        * Level 5 was the first run, with an earlier notebook; its σ is inferred. Level 0 has no saved OCR, and its PSNR/SSIM
        were recomputed from the saved outputs (the same recomputation matches Levels 1–9 to within 0.003 dB).
      </p>
      <div className="two" style={{ marginTop: 24 }}>
        <Figure src={img('restormer/gains_fine_tuned_vs_blurred.png')} caption="Gains of the fine-tuned model over the blurred input at every level." />
        <Figure src={img('restormer/char_acc_distribution.png')} caption="Spread of per-page character accuracy across the 40 test pages." />
      </div>
    </>
  )
}

function OcrSection() {
  const ocrLines = METHODS.map((m) => ({ key: m.key, name: m.label, color: m.color, dashed: m.key === 'gt' || m.key === 'input' }))
  return (
    <>
      <div className="two">
        <div className="panel">
          <h4>Character accuracy by level, %</h4>
          <Trend data={ocrSeries('char')} xKey="level" domain={[0, 100]} digits={1} series={ocrLines} />
          <p className="chart-note">From Level 5 EasyOCR reads nothing on the blurred page or the pretrained output. Fine-tuned stays at 96–98% through Level 5.</p>
        </div>
        <div className="panel">
          <h4>Token accuracy by level, %</h4>
          <Trend data={ocrSeries('token')} xKey="level" domain={[0, 100]} digits={1} series={ocrLines} />
          <p className="chart-note">Each token is a random 10-character string, so one wrong character loses it. Tokens fall much faster than characters from Level 6.</p>
        </div>
      </div>

      <div className="window">
        <div className="win-band skip"><small>Levels 0–1</small><b>Restoration unnecessary for OCR</b><span>The blurred page already reads at 98%; restoring it costs 5.7 points at Level 1.</span></div>
        <div className="win-band best"><small>Levels 2–5</small><b>OCR at 96–98%</b><span>Including pages OCR could not read at all; better than the sharp capture.</span></div>
        <div className="win-band part"><small>Levels 6–8</small><b>73–83% of characters</b><span>But only 22–32% of whole tokens.</span></div>
        <div className="win-band edge"><small>Level 9</small><b>52.8% of characters</b><span>Invented strokes and unstable training.</span></div>
      </div>

      <div className="table-scroll" style={{ marginTop: 24 }}>
        <table>
          <thead>
            <tr>
              <th>Level</th><th>Method</th><th className="num">Char acc</th><th className="num">Token acc</th>
              <th className="num">CER</th><th className="num">WER</th><th className="num">Exact match</th><th className="num">Empty OCR</th>
            </tr>
          </thead>
          <tbody>
            {OCR_LEVELS.flatMap((r) => METHODS.map((m, k) => {
              const x = r[m.key]
              return (
                <tr key={`${r.level}-${m.key}`} className={m.key === 'ft' ? 'hl' : ''} style={k === 0 ? { borderTop: '2px solid var(--rule)' } : undefined}>
                  <td>{k === 0 ? `L${r.level}` : ''}</td>
                  <td><span className="method-cell"><Swatch color={m.color} />{m.label}</span></td>
                  <td className="num">{x.char.toFixed(2)}%</td>
                  <td className="num">{x.token.toFixed(2)}%</td>
                  <td className="num">{x.cer.toFixed(2)}%</td>
                  <td className="num">{x.wer.toFixed(2)}%</td>
                  <td className="num">{x.exact.toFixed(1)}%</td>
                  <td className="num">{x.empty.toFixed(0)}%</td>
                </tr>
              )
            }))}
          </tbody>
        </table>
      </div>
      <p className="chart-note">
        EasyOCR (English, default models); character accuracy = max(0, 1 − Levenshtein / length of the true text);
        token accuracy = share of true tokens found in the transcript. CER and WER can exceed 100% when OCR inserts text.
      </p>
    </>
  )
}

function FontSection() {
  return (
    <div className="two" style={{ alignItems: 'start' }}>
      <div className="panel">
        <h4>Fine-tuned character accuracy by font, %</h4>
        <Trend
          data={fontSeries}
          xKey="level"
          domain={[30, 100]}
          digits={1}
          series={[
            { key: 'times', name: 'Times, fine-tuned', color: HEX.scratch },
            { key: 'verdana', name: 'Verdana, fine-tuned', color: HEX.naf },
            { key: 'gtTimes', name: 'Times, sharp', color: HEX.scratch, dashed: true },
            { key: 'gtVerdana', name: 'Verdana, sharp', color: HEX.naf, dashed: true },
          ]}
        />
        <p className="chart-note">From Level 6 the serif font degrades much faster: at Level 9, 34.3% of Times characters against 71.3% for Verdana.</p>
      </div>
      <Figure src={img('restormer/font_breakdown.png')} caption="PSNR, SSIM and OCR by font for every level and method." />
    </div>
  )
}

/* ---------------- Training and registration ---------------- */

function TrainingTable() {
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Level</th><th>Run</th><th className="num">σ (px)</th><th>Patch × batch</th><th className="num">Val PSNR @0</th>
            <th className="num">Best iter</th><th className="num">Val PSNR @best</th><th className="num">Val SSIM @best</th>
            <th className="num">Loss, first → last 50</th><th className="num">Minutes</th>
          </tr>
        </thead>
        <tbody>
          {L.map(({ level, train: t }) => (
            <tr key={level}>
              <td>L{level}</td>
              <td>{t.date}{t.inferred ? ' (legacy)' : ''}</td>
              <td className="num">{t.sigma}{t.inferred ? '*' : ''}</td>
              <td>{t.patch} px × {t.batch}</td>
              <td className="num">{t.valPsnr0.toFixed(2)}</td>
              <td className="num">{t.bestIter.toLocaleString('en-US')}</td>
              <td className="num">{t.bestValPsnr.toFixed(2)}</td>
              <td className="num">{t.bestValSsim.toFixed(4)}</td>
              <td className="num">{t.lossFirst.toFixed(4)} → {t.lossLast.toFixed(4)}</td>
              <td className="num">{t.minutes ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function RegistrationTable() {
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Level</th><th className="num">NCC before</th><th className="num">NCC after</th><th className="num">Worst pair</th>
            <th className="num">Scale x</th><th className="num">Scale y</th><th className="num">Shift x (px)</th><th className="num">Shift y (px)</th>
          </tr>
        </thead>
        <tbody>
          {L.map(({ level, train: t }) => (
            <tr key={level}>
              <td>L{level}</td>
              <td className="num">{t.nccBefore.toFixed(3)}</td>
              <td className="num"><b>{t.nccAfter.toFixed(3)}</b></td>
              <td className="num">{t.nccWorst.toFixed(3)}</td>
              <td className="num">{t.scaleX.toFixed(4)}</td>
              <td className="num">{t.scaleY.toFixed(4)}</td>
              <td className="num">{t.tx.toFixed(1)}</td>
              <td className="num">{t.ty.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const MODULES = [
  ['patch_embed', '3×3 conv, 3→48', '1,296', false],
  ['encoder_level1', '4 blocks, 48 ch, 1 head', '125,116', false],
  ['encoder_level2 (+ down1_2)', '6 blocks, 96 ch, 2 heads', '717,600', false],
  ['encoder_level3 (+ down2_3)', '6 blocks, 192 ch, 4 heads', '2,779,584', false],
  ['down3_4', '3×3 conv + PixelUnshuffle → 384 ch', '165,888', false],
  ['latent', '8 blocks, 384 ch, 8 heads', '14,370,448', true],
  ['up4_3 + reduce + decoder_level3', '6 blocks, 192 ch', '5,466,048', true],
  ['up3_2 + reduce + decoder_level2', '6 blocks, 96 ch', '1,389,216', true],
  ['up2_1 + decoder_level1', '4 blocks, 96 ch', '637,372', true],
  ['refinement + output', '4 blocks, 96 ch; 3×3 conv → 3, + input', '474,076', true],
]

/* ---------------- Phase 1 (superseded) ---------------- */

function Phase1() {
  return (
    <>
      <div className="two">
        <div className="panel">
          <h4>PSNR by focusStep level, dB</h4>
          <GroupedBars
            data={restormerFt}
            xKey="level"
            domain={[10, 26]}
            labels
            series={[
              { key: 'input', name: 'Blurred input', color: HEX.input },
              { key: 'pre', name: 'Pretrained', color: HEX.pre },
              { key: 'ft', name: 'Fine-tuned', color: HEX.ft },
            ]}
          />
        </div>
        <div className="panel">
          <h4>What Phase 1 was</h4>
          <p className="small muted">
            The first fine-tune used the small focusStep subset (Levels 1–4, about 113 pairs, 20 samples per font per level),
            no camera registration and a different split. It reached 24.88 dB at Level 1 (+3.21 dB) and lost 0.36 dB at
            Level 4. Phase 2 replaced it with registered pairs, a fixed 150/10/40 split and OCR on every level; these numbers
            are not comparable with Phase 2 and are kept for the record only.
          </p>
        </div>
      </div>
      <div style={{ marginTop: 24 }}>
        <LevelFigures
          levels={[1, 2, 3, 4].map((n) => ({
            label: `Level ${n}`,
            src: img(`restormer_ft_L${n}.png`),
            caption: `Phase 1, focusStep Level ${n}: blurred input, pretrained, fine-tuned and ground truth.`,
            meta: `Fine-tuned ${restormerFt[n - 1].ft.toFixed(2)} dB vs input ${restormerFt[n - 1].input.toFixed(2)} dB`,
          }))}
        />
      </div>
    </>
  )
}

/* ---------------- Page content ---------------- */

export default {
  lede:
    'Restormer computes attention across channels instead of pixels, so a full page fits in one pass. Phase 2 fine-tunes the natural-scene defocus checkpoint separately for every HDC 2021 level from 0 to 9, keeping the encoder frozen, and scores each model on 40 held-out pages with PSNR, SSIM and EasyOCR.',
  keyvals: [
    ['96.9%', 'OCR characters read at Level 5, from 0%', 'gain'],
    ['98.3%', 'Best OCR, Level 3 (blurred: 75.5%)', 'gain'],
    ['+2.4 to +8.4 dB', 'PSNR gain at every level, 0–9'],
    ['40 / 40', 'Pages improved in PSNR and SSIM, every level'],
  ],
  sections: [
    {
      id: 'overview',
      nav: 'Overview',
      title: 'Overview',
      lead: 'One model per blur level. Keep what the encoder learned about blur; retrain what the decoder makes of it.',
      body: (
        <div className="two" style={{ alignItems: 'start' }}>
          <div className="prose">
            <p>
              Off-the-shelf restoration weights do not transfer to text. Applied unchanged, the DPDD defocus
              checkpoint stays within 0.05 dB of the blurred input at every level and never helps OCR by more than
              4 points.
            </p>
            <p>
              Fine-tuned on 150 registered pairs per level, the same network improves PSNR and SSIM on every one
              of the 40 test pages at every level from 0 to 9. For OCR there is a clear window: no help at Levels
              0–1, where the page is already readable; 96–98% of characters at Levels 2–5, including pages OCR
              could not read at all; and a steady decline from Level 6 to 52.8% at Level 9.
            </p>
            <p className="small muted">
              This replaces the Phase 1 focusStep study, which used a smaller subset and a different split.
            </p>
          </div>
          <ol className="timeline">
            <li className="done"><b>Phase 1 · focusStep L1–4</b><span>Decoder fine-tune on a small subset. Superseded.</span></li>
            <li className="done"><b>Phase 2 · Level 5</b><span>The first per-level run: OCR from 0% to 96.9%.</span></li>
            <li className="done"><b>Phase 2 · Levels 0–9</b><span>One model per level with one notebook; every level evaluated on 40 pages.</span></li>
            <li><b>Level 10</b><span>Dataset ready, not trained.</span></li>
            <li><b>Combined Levels 5–10 model</b><span>Preprocessing done, training started, never evaluated.</span></li>
          </ol>
        </div>
      ),
    },
    {
      id: 'architecture',
      nav: 'Architecture',
      title: 'Architecture',
      lead: 'A four-level transformer encoder–decoder with 44 transformer blocks. The hatched blocks are frozen during fine-tuning.',
      body: (
        <>
          <ModelDiagram id="restormer" />
          <div className="two" style={{ marginTop: 24, alignItems: 'start' }}>
            <div className="panel">
              <h4>Where the parameters go</h4>
              <div className="split-bar" aria-label="3.79 M frozen, 22.34 M trainable">
                <i className="frozen" style={{ width: `${(3.79 / 26.13) * 100}%` }} />
                <i style={{ width: `${(22.34 / 26.13) * 100}%` }} />
              </div>
              <div className="split-legend">
                <span><i className="frozen" />Frozen encoder · 3,789,484</span>
                <span><i />Trainable · 22,337,160</span>
              </div>
              <div className="table-scroll" style={{ border: 0, marginTop: 12 }}>
                <table>
                  <thead><tr><th>Module</th><th>What it is</th><th className="num">Params</th></tr></thead>
                  <tbody>
                    {MODULES.map(([m, w, p, t]) => (
                      <tr key={m}>
                        <td><code>{m}</code></td>
                        <td style={{ whiteSpace: 'normal' }}>{w}</td>
                        <td className={`num ${t ? 'delta-pos' : ''}`}>{p}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="chart-note">The latent level alone holds 55% of the weights. Trainable modules in green; total 26,126,644.</p>
            </div>
            <div className="fig-row">
              <Figure src={img('restormer/restormer_architecture.png')} caption="Restormer in its defocus-deblurring configuration: dim 48, blocks [4, 6, 6, 8], 4 refinement blocks, heads [1, 2, 4, 8]." />
              <Figure src={img('restormer/transformer_block_detail.png')} caption="One transformer block: MDTA attends across channels; GDFN gates a depthwise-conv feed-forward." />
            </div>
          </div>
        </>
      ),
    },
    {
      id: 'training',
      nav: 'Training',
      title: 'Training recipe',
      lead: 'Identical for every level: only LEVEL changes in notebook 05.',
      body: (
        <>
          <Figure src={img('restormer/pipeline_overview.png')} caption="From the paired TIFFs to the per-level metrics: load, split, estimate blur, register, fine-tune, infer, score." />
          <div className="two" style={{ marginTop: 24, alignItems: 'start' }}>
            <div className="panel">
              <h4>Recipe</h4>
              <Setup
                rows={[
                  ['Data', '200 pairs per level; 150 train / 10 val / 40 test, same sample IDs at every level (seed 42)'],
                  ['Loading', 'TIFF → mean over channels → divide by the 99.5th percentile → clip; 48 px trim after registration'],
                  ['Blur estimate', 'Gaussian σ ∈ {5, 10, …, 100} px template-matched against the real input; median of 4 pairs'],
                  ['Model', 'DPDD defocus checkpoint, loaded strictly; encoder path frozen'],
                  ['Patches', 'Text-containing crops, rotations and flips; 128 px × 4 up to σ = 30, 192 px × 2 above'],
                  ['Optimiser', 'AdamW, LR 2×10⁻⁵, weight decay 10⁻⁴, cosine to 10⁻⁷ over 3,000 iterations'],
                  ['Loss', 'Charbonnier (ε = 10⁻³), AMP fp16, gradient clip 1.0'],
                  ['Selection', 'Validation every 500 iterations; best validation SSIM kept'],
                  ['Inference', '768 px tiles, 192 px overlap (256 above σ = 50), feathered blending'],
                  ['Compute', 'Google Colab, Tesla T4'],
                ]}
              />
            </div>
            <Figure src={img('restormer/training_curves_all_levels.png')} caption="Training loss and validation PSNR/SSIM for every level." />
          </div>
          <h3 style={{ margin: '32px 0 14px' }}>Per-level training runs</h3>
          <TrainingTable />
          <p className="chart-note">* Level 5 is the legacy run: no training log, and σ and patch size are inferred from the notebook.</p>
        </>
      ),
    },
    {
      id: 'registration',
      nav: 'Registration',
      title: 'Registering the two cameras',
      lead: 'The sharp and defocused captures do not line up exactly. One global affine per level, fitted on 12 training pairs, warps every ground-truth image into the input frame.',
      body: (
        <>
          <div className="two" style={{ alignItems: 'start' }}>
            <RegistrationTable />
            <div className="panel">
              <h4>How it is fitted</h4>
              <ul className="facts small">
                <li>320 × 160 px templates from the blurred ground truth, on an 80 px grid, searched ±60 px in the input.</li>
                <li>Matches with NCC ≥ 0.8 (relaxed to 0.7 or 0.6 when fewer than 50 are found) feed a RANSAC affine.</li>
                <li>Alignment (NCC of blurred ground truth against the input) is logged for all 200 pairs in <code>pairing_checks.csv</code>.</li>
                <li>Alignment falls from 0.97 at Level 1 to 0.78 at Level 9, which caps both the training signal and the measured PSNR at heavy blur.</li>
              </ul>
            </div>
          </div>
          <div className="fig-row" style={{ marginTop: 24 }}>
            <Figure src={img('restormer/psf_by_level.jpg')} caption="The point-target images shipped with each step: the sharp dot stays fixed while the defocused disc widens." />
            <Figure src={img('restormer/registration_by_level.png')} caption="Alignment before and after registration, and the fitted scale and shift, by level." />
          </div>
        </>
      ),
    },
    {
      id: 'explorer',
      nav: 'Level by level',
      title: 'Level by level',
      lead: 'Pick a level for its numbers, what happens there, and four test pages side by side.',
      body: <LevelExplorer />,
    },
    {
      id: 'results',
      nav: 'Image quality',
      title: 'Image quality across Levels 0–9',
      lead: 'Full-image PSNR and SSIM against the registered sharp capture, 40 test pages per level.',
      body: <AllLevels />,
    },
    {
      id: 'ocr',
      nav: 'OCR',
      title: 'OCR: can the page be read?',
      lead: 'The primary outcome. EasyOCR on every test page, for the blurred input, the pretrained and fine-tuned outputs, and the sharp capture.',
      body: <OcrSection />,
    },
    {
      id: 'fonts',
      nav: 'Fonts',
      title: 'Serif against sans-serif',
      body: <FontSection />,
    },
    {
      id: 'phase1',
      nav: 'Phase 1',
      title: 'Earlier: Phase 1 on focusStep (superseded)',
      body: <Phase1 />,
    },
    {
      id: 'takeaways',
      nav: 'Takeaways',
      title: 'Takeaways',
      body: (
        <Takeaways
          shows={[
            'Fine-tuning, not the architecture alone, produces the gain: the pretrained model is never more than 0.05 dB above the input.',
            'OCR, not PSNR, is the outcome that matters: the highest PSNR (Level 1) comes with an OCR loss, and the largest OCR gain (Level 5) with a moderate PSNR.',
            'There is a usefulness window: Levels 2–5 reach 96–98% of characters; Levels 0–1 need no restoration.',
            'The sharp capture is not a ceiling: restored pages at Levels 2–5 read better than it.',
          ]}
          limits={[
            'Each model is trained and tested on its own level, so it is level-matched rather than blind; cross-level generalisation is untested.',
            'One run and one seed per level; spreads are over the 40 test pages, not over trainings.',
            'Confounds across levels: registration quality falls, patch size changes at Level 7, σ for Levels 0–2 is a floor, and Level 6 used 128 px patches.',
            'OCR rests on EasyOCR alone, and Level 0 has no OCR.',
          ]}
          next={[
            'Run the Level 0 OCR cells, re-run Level 6 with 192 px patches and Level 9 for longer.',
            'Train Level 10 and finish and evaluate the combined Levels 5–10 model.',
            'Evaluate every checkpoint across levels, and add a second recognizer such as TrOCR.',
          ]}
        />
      ),
    },
  ],
}
