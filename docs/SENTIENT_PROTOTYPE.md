# Sentient: the model, the simulation, and what it shows

Working document for the Sentient prototype in this repository. It describes the current model
(`sentient/rich/`), how it is validated, what the results support, what is protected and what
is shared. The first prototype (`sentient/`, two factors, four channels) is kept for reference
and summarised at the end. Last full update: 2026-09-24.

## 1. What Sentient is

An intelligence layer for physical spaces. In one space it does four things, continuously:

1. **Sense.** A few off-the-shelf sensors report what can be observed: how many people, how
   they move, sound level and character, temperature, CO2, and, only if a guest opts in, their
   own wearable.
2. **Infer.** The model treats what matters, who is there, what state they are in, what they
   are there to do, and whether the building's own equipment is working, as hidden. It holds a
   probability over those states and updates it every few seconds. It never outputs "she is
   stressed". It outputs "seventy percent stressed, given agitated movement, heart rate above
   baseline and low HRV".
3. **Recommend.** One action or none, to a person: dim or raise the lights, lower or raise the
   sound, cool or warm, ventilate, alert staff, raise a maintenance ticket. Every
   recommendation carries the inferred state, the reason, a confidence, and a forecast of what
   every sensor should read a few minutes later.
4. **Score.** Forecasts are sealed with a hash when written and marked as hit or miss when the
   readings arrive. Nothing is edited after the fact.

Copilot first. A person accepts or dismisses every recommendation; dismissals raise the model's
bar for that action. Autonomy only where staff have accepted enough, and always explainable
and reversible.

**Affect is the core; energy is the wedge.** Energy against a preset is the line an owner can
check on a bill in month one. Inferring human state, and acting on it with a person when a
thermostat cannot help, is what Sentient is. The evaluation scores both.

### Where it stands

Designed, implemented, and validated in a simulation deliberately richer than the model. No
real sensors and no real people yet. Sensors for a first real room are the next step; the same
code runs on the simulated feed and on a Home Assistant feed (`sentient/sensors.py`).

Two conversations shape the commercial path:

- **Driftwood Hospitality** (2026-09-08; Carlos Junior, ~90 hotels, in-house operating system,
  new brand concept "Hipsip" at a Fort Lauderdale Hilton). Declined to be the capital; offered
  lab runway for Sentient inside Hipsip; asked for a technical session led by Mahault. The
  shape we want: co-branded, exclusive in Driftwood's markets for a period, Driftwood owns
  Hipsip and its property data, Sentient owns the model and all improvements and keeps the
  other verticals. Not "a division of Driftwood".
- **McPike Global Family Office** (2026-09-24; Drew Colaiezzi, Dinesh Choudhary; ex-quant, own
  capital, seed to Series C). Engaged on inferred state, nested spaces, and fault detection
  falling out of forecast scoring. Asked for financials, a competitive matrix and a data room.
  No technical session booked yet. Follow-up materials in `C:/dev/projects/sentient/mcpike-followup/`.

Closest comparator: **PassiveLogic** ($154M raised; $74M Series C, September 2025; Brookfield,
Johnson Controls and NVIDIA's venture arm on the cap table). Physics-based digital twin plus
autonomous HVAC and energy control. It models the building and not the people. Sentient's
framing is the people layer on top of a controller like that, not a competitor to it.

## 2. IP posture

The idea, "spaces that adapt to guests", is not protectable. What is hard to copy is the model,
the evaluation loop, and the data it generates. The realistic risk is absorption via runway
paid for engineering in a partner's building, not cloning. Protection, in order:

1. **Paper before runway.** The deal clauses above, set as requirements before terms exist.
2. **Mutual NDA before any technical session**, covering the other side's technical people by
   name.
3. **Share the what and the why, keep the how** (below).
4. **Being the people who can build it.** For the next year the moat is execution speed.
5. **Provisional patent** on closed-loop, prediction-scored environmental control with explicit
   consent and revoke, filed before a deep-dive. Not legal advice; the standard move.
6. **Sentient IP between founders written down** before a third party enters.

**Share:** block-diagram architecture; the loop; the three success numbers and the baseline;
the sensor list and cost; simulation results including calibration and affect scores; the
safety design (operator intent, guest purpose, do-nothing default, override and learn).

**Keep:** the generative model structure (factors, modalities, dependencies, the A/B/C matrices
and priors); inference and policy code; the evaluation methodology in detail; the demand-engine
internals; anything that lets a competent team reproduce the model rather than be convinced by
its behaviour.

## 3. Why a simulation, and why it is richer than the model

A running loop convinces where a diagram does not. A simulation lets all the numbers be shown
before any hardware, lets objections from calls become scenarios that run live, and is the
honest shape of the technology: active inference is a generative model of the space, and the
simulator is that model's world.

The simulator is deliberately richer than the model. If they shared a structure, "it works"
would prove nothing. The world (`sentient/rich/space.py`) has:

- Individual occupants with a purpose, baseline arousal, sensitivities to sound, light,
  crowding, warmth and stale air, sociability, a **valence** the model never sees, a wearable
  or not, an expected stay, and optionally a mood the room cannot fix.
- Poisson group arrivals by hour, per-person departures, a capacity.
- Temperature with inertia (cooling is felt about twenty minutes after the command), CO2 that
  builds with people and clears with ventilation, a cooling unit that can silently degrade,
  staff who take a variable time to respond, daylight that moves with the clock.
- Sensor artifacts: radar misses and ghosts, stale frames, a microphone that is a power sum of
  floor, music, air handling, conversation and door slams (and can be configured dead),
  wearables that drop out of sync.
- Simulated staff who accept or dismiss recommendations and know things the model does not,
  such as whether the people are enjoying themselves.

One step is two minutes. Every run also logs the world's hidden arousal and mood so the
inference itself can be scored (section 6).

## 4. The model

`sentient/rich/spec.py` holds the model as data; `sentient/rich/agent.py` runs it in pymdp.

### Hidden state (eight factors)

| Factor    | States                                      | Seen through                                    |
|-----------|---------------------------------------------|-------------------------------------------------|
| occupancy | empty, few, some, crowded                   | radar count, door entries, CO2, wearables       |
| arousal   | calm, elevated, stressed                    | motion, sound level, heart rate, HRV            |
| valence   | negative, neutral, positive                 | sound character (laughter vs hum), HRV          |
| activity  | resting, working, socializing, transiting   | dwell time, motion, sound character             |
| thermal   | cold, comfortable, warm                     | temperature                                     |
| air       | fresh, stale                                | CO2 given occupancy                             |
| hvac      | ok, degraded                                | how temperature responds to cooling commands    |
| music     | off, low, high                              | sound level given people; moved by the actions  |

Operator **intent** (restore, work, elevated, service) and guest **purpose** are not inferred.
Humans set them; they select the preferences the planner scores against. The same loud room
is a problem under "restore" and the goal under "elevated".

### Observations (ten channels)

radar count · door entries · motion · sound level · sound character · temperature · CO2 ·
heart rate · HRV · dwell time. Each depends on a subset of factors; each factor is seen through
more than one channel. "Unknown" carries mass on the wearable channels: most guests do not opt
in, and devices drop sync.

### Actions (ten)

do nothing · dim lights · raise lights · lower sound · raise sound · cool air · warm air ·
ventilate · staff alert · maintenance ticket. Doing nothing has a prior.

### Transitions that depend on other factors

- **thermal depends on hvac**: cooling moves a warm room toward comfortable only if the unit
  works. This is the mechanism behind fault inference.
- **air depends on occupancy**: stale rises with people; ventilate clears it.
- **valence depends on arousal and thermal**: stress and discomfort sour a mood.
- **arousal depends on valence**: a staff visit calms someone who is stressed *and not enjoying
  it*, and does nothing for an excited happy group. Without this the planner paged parties.
- **occupancy under "staff alert" is intent-specific**: it thins a queue in a service space
  (a lobby) and leaves a meeting room alone. Without this the planner paged staff to improve
  CO2.
- **music** is moved deterministically by raise/lower sound and observed through sound level.
  Without it the model attributed all sound to people, was surprised by a calm guest with loud
  music every step, and flagged its own microphone.

### Inference details worth knowing

- pymdp (JAX, 1.x) with `A_dependencies` and `B_dependencies`. Observations are passed as
  integer indices per modality (the default `categorical_obs=False`); passing one-hot arrays
  silently produces wrong posteriors. The older agents in `services/active_inference/` still do
  that and are flagged for a separate fix.
- pymdp's one-step filter does **not** pass "we commanded cooling and the temperature did not
  move" back from the thermal factor to the hvac factor. That single message is computed
  explicitly from the same A and B (`RichAgent._update_hvac`) and fed back as the hvac prior.
  The sustained control state of the unit (cooling on, heating on) is a known input for that
  message, like intent.
- Policies are single-step; preferences are on observations; planning uses expected free
  energy with gamma 8 and a policy prior that favours doing nothing.

### Safety mechanisms, each tested

- **Do-nothing default.** An action is recommended only when its posterior probability beats
  doing nothing by a margin. Actions are walked in preference order, so a dismissed lever gives
  way to the next one.
- **Lever mask.** An action already at its floor (lights already dim, sound already low,
  cooling already on, staff just alerted) is unavailable. No lever, no recommendation.
- **Override learning.** A dismissal raises that action's margin by 0.15 (cap 0.95); an
  acceptance lowers it slightly. A dismissed setback raises how long the room must look empty
  before the next one.
- **Two labelled rules on top of the model:** a confidently and persistently empty room gets a
  setback; a room believed to have a degraded HVAC unit while uncomfortable gets a maintenance
  ticket instead of another command.
- **Every recommendation carries** the state, the reason, a confidence, a primary sealed
  forecast, and a forecast for every channel.

## 5. What is measured

Per run, from the append-only ledger (`sentient/ledger.py`, same sealing rule as the platform's
Signal Ledger):

1. **Forecast accuracy** of the primary sealed forecast three steps ahead, against a persistence
   baseline ("same as now"). Dismissed recommendations are recorded but not scored: their
   forecast assumed an action that never happened.
2. **Override rate** over time (dismissed / recommendations per third of the run).
3. **Energy** proxy (lighting plus cooling plus heating plus ventilation) against a fixed
   "bright and cooled" preset.
4. **Calibration**: reliability bins and expected calibration error of the primary forecasts.
5. **Per-channel forecast accuracy** and a **sensor flag** when one channel's forecasts collapse
   while persistence would have been right and the other channels hold.
6. **Affect inference**: correlation and rank AUC between the model's P(stressed) and the
   world's hidden mean arousal, and between P(positive) and hidden mean mood.

## 6. Scenarios and results

Seed 0, 60 steps of two minutes. Eleven scenarios: the four from the Driftwood call (including
Carlos's two objections), three where the model must cope with being wrong, and four that only
a richer model can handle.

| Scenario        | Set-up                                                    | What the model did                                                                 | Forecast acc / baseline | Energy vs preset | r, stress vs true arousal |
|-----------------|-----------------------------------------------------------|------------------------------------------------------------------------------------|-------------------------|------------------|---------------------------|
| restore_pod     | one guest, bright, loud music, warm                       | lowered the sound twice; arousal 0.62 to 0.31                                      | 0.98 / 0.93             | 45% less         | 0.68                      |
| bar_9pm         | intent elevated, groups arriving all evening              | held 58 of 60; one raise-sound, one ventilate                                       | 1.00 / 0.98             | 8% less          | 0.65                      |
| flat_bar        | intent elevated, music off, three people                  | raised the sound three times, all accepted; mood 0.46 to 0.67                      | 0.98 / 0.95             | 7% more          | 0.78                      |
| ambiguous_hr    | resting guest, elevated HR, room already dim, quiet, cool | nothing, 60 of 60                                                                  | 0.98 / 0.96             | 30% less         | 0.67                      |
| lobby_peak      | check-in wave                                             | seven staff alerts; set the lobby back once empty                                  | 0.40 / 0.51             | 13% less         | 0.34                      |
| happy_group *   | five delighted friends, lounge set to restore             | tried to hush, dismissed five times in the first third, raised its bar, stopped    | 0.38 / 0.64             | 27% less         | 0.74                      |
| stale_meeting   | six people working, CO2 climbing                          | ventilated eight times; CO2 peaked at 788 vs 1,516                                  | 0.78 / 0.98             | 2% less          | 0.48                      |
| hvac_failure    | warm room, cooling unit dies at step 8                    | commanded cooling, watched it fail, P(degraded) 0.05 to 0.99, ticket at step 18, repaired, room at 21.5 | 0.74 / 0.54 | 33% less | 0.56                  |
| cold_morning    | work room at 8am, 4 degrees outside                       | warmed once; room at 23.7                                                          | 0.79 / 0.85             | 16% less         | 0.79                      |
| stressed_guest  | one distressed guest, room already comfortable            | read stress on the first frame, called a person; mood 0.31 to 0.67 (preset 0.27)   | 0.98 / 0.95             | 30% less         | 0.64                      |
| sensor_failure  | microphone dead, reads constant quiet                     | sound channel flagged; other channels contradict it                                | 0.79 / 0.83             | 25% less         | 0.48                      |

\* The happy_group row is from the last complete full run, which predates the final valence
dependency in the arousal transition. With it, the test suite asserts at most ten hushing
attempts and at most two recommendations in the last third; that run was not repeated because
the machine was low on memory. Rerun the set to refresh the row.

Affect inference, where mood varies enough to score it: AUC positive 0.84 (flat_bar), 0.89
(hvac_failure), 0.71 (happy_group), 0.68 (stressed_guest); AUC negative 0.92 (stressed_guest).
AUC stressed 1.00 (restore_pod, flat_bar, ambiguous_hr), 0.88 (sensor_failure), 0.78
(stressed_guest), 0.73 (happy_group).

### How to read these

- **The two objections from the Driftwood call behave.** The bar reads as loud and agitated and
  the model still holds, because that is the intent. The ambiguous case has no lever and a live
  wearable saying someone is there, so no recommendation and no setback.
- **Fault detection is model-based.** The unit dies silently; the belief that it is degraded
  climbs as commanded cooling fails to move the temperature; a ticket goes out instead of
  another command. Nobody wrote a detector.
- **A person, not a thermostat.** The distressed guest is the affect story: nothing in the room
  would help, the model read stress and unhappiness from behaviour and physiology, and called
  staff. Ambiguous HR, the same sensors with a happy resting guest, gets nothing.
- **Forecasts beat persistence where the model acts** and match it in settled rooms. Where the
  world changes in ways the model does not yet capture (ventilation speed, a lobby's turnover)
  persistence wins, and the report says so.
- **Sensor health falls out of forecasting, given redundancy.** With activity as a latent, a
  quiet microphone can be explained away as resting people. It is caught because heart rate
  and count contradict that story. A sensor that is the only witness to a state cannot be
  caught this way.
- **Calibration** is good where things happen (ECE 0.06 to 0.2) and underconfident in calm
  rooms (the model says 50% and is right 95% of the time). The observation model is soft by
  design; the first real room is the data to learn it from.

## 7. What was learned building it

- **Activity must not flatten the arousal signal.** An early observation model let "resting"
  shift sound and motion so far that a loud reading was equally surprising under every arousal
  level and told the model nothing.
- **Levers the system controls belong in the hidden state.** See music.
- **Dynamics are intent-specific.** See staff alert.
- **Affect distinctions are what make actions safe.** A staff visit that helped anyone read as
  stressed paged the happy group eleven times. Conditioning on mood fixed it.
- **A richer model explains more away.** Fault detection needs redundant channels.
- **Score only what was applied.** A dismissed recommendation's forecast says nothing about the
  model or the sensor.
- **Flag sensors relative to the other channels**, not against a fixed threshold.
- **Thresholds were behind half the bad runs**: what counts as loud, warm, stale. Learn them from
  the real room.
- **Renormalised preferences need a high margin cap.** With most levers masked, the surviving
  action's probability can exceed a 0.6 cap; dismissals now raise margins to 0.95.

## 8. Known gaps and next steps

- Calm-room forecasts are underconfident; the ventilate forecast is optimistic at a three-step
  horizon; motion at a packed bar is mis-modelled. All are observation-model parameters to
  learn from data (`infer_parameters` is off).
- Mood inference is undefined where a scenario's mood barely moves; the scoring reports that
  rather than inventing a number.
- No multi-space coordination. The nesting story (room, floor, building, portfolio) is design.
- The Brandclave demand signal as an input to the space model is the stage-three
  differentiator. Interface designed, not built.
- Next: sensors in one real room (Aqara FP2 radar ~$80, Airthings View Plus or Netatmo ambient
  ~$200, gateway ~$150, guest wearable via vendor API); same code on both feeds; learn the
  observation model; then the technical sessions.

## 9. Running it

```
python -m sentient.rich.run --scenario all --steps 60 --seed 0     # ~2 s/step, ~25 min
python -m sentient.rich.run --scenario hvac_failure                  # one scenario
pytest tests/test_sentient_rich.py                                   # 14 tests, ~5 min
```

Ledgers (`<scenario>.jsonl`) and `results.json` land in `data/sentient_rich/` unless `--out`
is given. The first prototype runs the same way from `sentient.run` and `tests/test_sentient_sim.py`.

Evidence pack for the deck (nine figures, results table, what we can and cannot claim):
https://claude.ai/artifact/1NDQqRRykL2Mm9kec9dUPX

## 9b. October 2026: a rulebook comparator, model fixes, held-out results, live demo

**Comparator.** `sentient/rich/rules.py` is what a competent integrator would program on the same
sensors: fixed thresholds, operator intent, and a hand-written detector for a failing cooling unit.
`sentient/rich/evaluate.py` runs Sentient, the rulebook and the fixed preset on the same world, seed
and simulated staff, 11 scenarios x 10 seeds, with 95% intervals, and resumes after interruption.

**Model changes** (tuned on seeds 0-9, then frozen):

- A ninth factor, `lighting` (dim, mid, bright), controlled like `music` and seen through an 11th
  channel, `lux` (the radar's own light sensor; "unknown" when absent). Preferences on lux per intent.
- `warm` starts at 24 C, the same comfort band the rulebook uses (was 25).
- Arousal transitions are more persistent; motion forecasts in busy rooms stopped failing.
- CO2: under 800 is fresh (was 700); no preference for fresh over ok; slower stale dynamics.
- Setback needs the radar and the door counter both at zero, a longer streak in a service space,
  and a dismissed setback resets the streak.
- In a service space a staff call needs a queue that is there (P(some or crowded) > 0.6).
- Live channel health in the agent: a channel is flagged on a flatline (same raw value four reads
  running) or when its forecasts fail while "same as now" is right at least 95% of the time and the
  other channels hold. A flagged channel still feeds inference; actions that only it could confirm
  are masked. An earlier version replaced flagged readings with the model's own expectation; that
  fed back on itself and powered down occupied rooms, so it was removed.

**Held-out results** (seeds 10-19, never used for tuning; rulebook unchanged; `data/sentient_eval_heldout/`):

- Against the rulebook, 33 comparisons (mood, energy, dismissed calls x 11 rooms): Sentient ahead 8,
  level 23, rulebook ahead 2 (before the changes, on seeds 0-9: 7, 19, 7). Rulebook leads: lobby
  dismissed calls 1.2 vs 0 a session; meeting-room energy +2% (Sentient keeps the air fresher, mood
  0.59 vs 0.56).
- Mood above the preset in 10 of 11 rooms, level in the happy group; distressed guest 0.63 vs 0.25.
- Failing cooling unit ticketed first in 10 of 10 runs, about 17 min vs 28 after failure; one false
  ticket in 110 runs.
- Forecasts beat or match persistence in 85 of 110 runs (was 80); still 0 of 10 in the stale meeting.
- Dead microphone flagged live in 10 of 10 runs within six minutes; healthy rooms raise a warning in
  32 of 100 runs, mostly wearables losing sync, none on a channel any action depends on.
- Stress detection AUC 0.88-0.99 in five of eight rooms where stress occurs; 0.42 in the lobby rush.
- Known quirk: the cold-morning room starts at exactly 24 C, so the model cools it first and then
  holds about 21.8 C (the rulebook does the same). The test now checks the room ends comfortable.
- 0.54 s per decision on a laptop, one decision every two minutes.

**Live demo.** `python -m sentient.rich.demo`, then http://127.0.0.1:8765. The real engine on a
simulated room, Sentient, rulebook and preset side by side with a live scoreboard; buttons to fail
the cooling unit, kill the microphone, add a distressed guest or a group, empty the room, change
the room's intent; a mode where the viewer plays the staff and accepts or dismisses; a toggle that
shows the hidden truth; and the sealed record with Verify and Tamper. Runs locally only.

## 10. History: the first prototype (2026-09-08)

`sentient/` is the two-factor version built the evening after the Driftwood call and kept as a
reference. Hidden state: occupancy (empty, some, crowded) and arousal (calm, elevated,
stressed). Four channels: radar count, motion, sound level, opt-in wearable. Five actions.
Seven scenarios. Its results (seed 0, 60 steps): forecasts beat persistence in six of seven;
the bar and ambiguous cases held; the happy group was hushed eleven times before dismissals
stopped it; a dead microphone was flagged with no detector written (it had no activity latent
to hide behind); 80% less energy than the preset in the restore pod. Its sixteen tests still
pass. Everything it could do, the rich model does with more structure, and the McPike-era
materials that described it have been updated to the rich model.
