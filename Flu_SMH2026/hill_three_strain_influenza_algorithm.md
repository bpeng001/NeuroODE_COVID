# Three-Strain Hill-Style Influenza Model

This document describes a simplified reproduction of the Hill et al. multi-strain influenza model using three circulating groups:

- H1N1
- H3N2
- Influenza B

The model retains Hill's joint-population structure, current-season vaccination, previous-season exposure-history susceptibility modifiers, and subtype-specific transmission. The two influenza B lineages are collapsed into one B category.

---

## 1. Model Objective

Construct a deterministic, non-age-structured, three-strain SEIR model beginning with the 2009/10 or 2010/11 influenza season.

The main subtype-specific outputs are

$$
Z_{H1,t}, \qquad Z_{H3,t}, \qquad Z_{B,t},
$$

with total influenza incidence

$$
Z_{\mathrm{flu},t}
=
Z_{H1,t}
+
Z_{H3,t}
+
Z_{B,t}.
$$

The model should ultimately allow estimation of three subtype-specific transmission parameters:

$$
\beta_{H1}, \qquad \beta_{H3}, \qquad \beta_B.
$$

---

## 2. Model Inputs

Use:

- weekly influenza-like illness (ILI) or influenza-attributable ILI;
- weekly influenza positivity;
- weekly H1/H3/B subtype proportions or counts;
- seasonal or time-varying vaccination uptake;
- subtype-specific vaccine effectiveness;
- total population size;
- simulation start date and influenza-season boundaries.

Use

$$
m \in \{H1,H3,B\}.
$$

---

# Algorithm

## Algorithm 1. Initialize the model

1. Set the simulation time step:

$$
\Delta t = 1 \text{ day}.
$$

2. Prefer population proportions for initial implementation:

$$
N = 1.
$$

3. Define previous-season exposure-history groups:

$$
h \in
\{
N,H1,H3,B,V,H1V,H3V,BV
\}.
$$

where:

- $N$: no influenza infection and no vaccination in the previous season;
- $H1$: H1 infection in the previous season;
- $H3$: H3 infection in the previous season;
- $B$: influenza B infection in the previous season;
- $V$: vaccinated but not infected;
- $H1V$: H1 infection plus vaccination;
- $H3V$: H3 infection plus vaccination;
- $BV$: B infection plus vaccination.

4. Define current-season vaccination status:

$$
X \in \{N,V\},
$$

where $N$ is currently unvaccinated and $V$ is currently vaccinated.

5. For each exposure-history group $h$, define susceptible states:

$$
S^{N,h}, \qquad S^{V,h}.
$$

6. For each strain $m$, define:

$$
E_m^N,\ I_m^N,\ R_m^N,
$$

and

$$
E_m^V,\ I_m^V,\ R_m^V.
$$

---

## Algorithm 2. Construct the susceptibility matrix $f(h,m)$

Define

$$
f(h,m)
=
\text{relative susceptibility to current strain }m
\text{ given previous-season history }h.
$$

Use:

| Previous-season history $h$ | H1 | H3 | B |
|---|---:|---:|---:|
| $N$ | $1$ | $1$ | $1$ |
| $H1$ | $a$ | $1$ | $1$ |
| $H3$ | $1$ | $a$ | $1$ |
| $B$ | $1$ | $1$ | $a$ |
| $V$ | $c_{H1}$ | $c_{H3}$ | $c_B$ |
| $H1V$ | $\min(a,c_{H1})$ | $c_{H3}$ | $c_B$ |
| $H3V$ | $c_{H1}$ | $\min(a,c_{H3})$ | $c_B$ |
| $BV$ | $c_{H1}$ | $c_{H3}$ | $\min(a,c_B)$ |

Set

$$
a = 0.7883.
$$

This means previous infection with the same strain leaves about 78.8% residual susceptibility in the next season.

Equivalent protection is

$$
1-a = 0.2117.
$$

Residual vaccine-derived susceptibility is

$$
c_{m,y}
=
1-\xi \alpha_{m,y-1},
$$

where:

- $\alpha_{m,y-1}$ is previous-season vaccine effectiveness against strain $m$;
- $\xi$ is the fraction of previous-season vaccine effectiveness carried into the next season.

Use

$$
\xi = 0.0051.
$$

Because influenza B is collapsed into one category, omit Hill's B-lineage cross-reactivity parameter $b$.

---

## Algorithm 3. Apply current-season vaccination

At each day $t$, move individuals from unvaccinated to vaccinated states according to vaccination rate $\mu_y(t)$.

Conceptually:

$$
S^{N,h} \rightarrow S^{V,h},
$$

$$
E_m^N \rightarrow E_m^V,
$$

$$
I_m^N \rightarrow I_m^V,
$$

$$
R_m^N \rightarrow R_m^V.
$$

Vaccination does not erase previous-season exposure history.

---

## Algorithm 4. Calculate subtype-specific force of infection

For each strain:

$$
\lambda_{H1}(t)
=
\beta_{H1}
\left[
I_{H1}^{N}(t)+I_{H1}^{V}(t)
\right],
$$

$$
\lambda_{H3}(t)
=
\beta_{H3}
\left[
I_{H3}^{N}(t)+I_{H3}^{V}(t)
\right],
$$

$$
\lambda_B(t)
=
\beta_B
\left[
I_B^{N}(t)+I_B^{V}(t)
\right].
$$

If counts rather than proportions are used, divide the infectious term by $N$:

$$
\lambda_m(t)
=
\beta_m
\frac{I_m^N(t)+I_m^V(t)}{N}.
$$

---

## Algorithm 5. Calculate infection flow from each exposure-history group

For unvaccinated individuals:

$$
F_{m,h}^{N}(t)
=
f(h,m)
S^{N,h}(t)
\lambda_m(t).
$$

For vaccinated individuals:

$$
F_{m,h}^{V}(t)
=
f(h,m)
S^{V,h}(t)
\left(1-\alpha_{m,y}\right)
\lambda_m(t).
$$

The current-season vaccine is therefore modeled as leaky protection through

$$
1-\alpha_{m,y}.
$$

Do not sum $f(h,m)$ values by themselves.

For each exposure-history group, the total infection removal rate is

$$
\sum_m F_{m,h}^{X}(t).
$$

---

## Algorithm 6. Update susceptible states

For every previous-season history group $h$:

$$
\frac{dS^{N,h}}{dt}
=
-
\sum_m
F_{m,h}^{N}
-
\mu_y(t)S^{N,h}.
$$

For vaccinated susceptible individuals:

$$
\frac{dS^{V,h}}{dt}
=
-
\sum_m
F_{m,h}^{V}
+
\mu_y(t)S^{N,h}.
$$

For the first reproduction, demographic birth/death terms may be omitted.

---

## Algorithm 7. Update subtype-specific $E$, $I$, and $R$

First aggregate infection flow over exposure-history groups:

$$
F_m^N(t)
=
\sum_h
F_{m,h}^N(t),
$$

$$
F_m^V(t)
=
\sum_h
F_{m,h}^V(t).
$$

For unvaccinated individuals:

$$
\frac{dE_m^N}{dt}
=
F_m^N
-
\gamma_{1,m}E_m^N,
$$

$$
\frac{dI_m^N}{dt}
=
\gamma_{1,m}E_m^N
-
\gamma_2I_m^N,
$$

$$
\frac{dR_m^N}{dt}
=
\gamma_2I_m^N.
$$

For vaccinated individuals:

$$
\frac{dE_m^V}{dt}
=
F_m^V
-
\gamma_{1,m}E_m^V,
$$

$$
\frac{dI_m^V}{dt}
=
\gamma_{1,m}E_m^V
-
\gamma_2I_m^V,
$$

$$
\frac{dR_m^V}{dt}
=
\gamma_2I_m^V.
$$

---

## Algorithm 8. Record subtype-specific incidence

For each subtype:

$$
C_m(t)
=
F_m^N(t)
+
F_m^V(t).
$$

Aggregate daily incidence to week $w$:

$$
Z_{m,w}
=
\sum_{t\in w}
C_m(t)\Delta t.
$$

This gives:

$$
Z_{H1,w},
\qquad
Z_{H3,w},
\qquad
Z_{B,w}.
$$

Total weekly influenza incidence is

$$
Z_{\mathrm{flu},w}
=
Z_{H1,w}
+
Z_{H3,w}
+
Z_{B,w}.
$$

---

## Algorithm 9. Observation model

Use a season-specific ascertainment probability:

$$
\widehat Y_{m,w}
=
\epsilon_y Z_{m,w}.
$$

Compare:

$$
\widehat Y_{H1,w},
\qquad
\widehat Y_{H3,w},
\qquad
\widehat Y_{B,w}
$$

with observed subtype-specific influenza-attributable ILI.

For the initial reproduction, a placeholder value can be used.

For actual fitting, $\epsilon_y$ should be estimated or otherwise calibrated to the surveillance system.

---

## Algorithm 10. End-of-season exposure-history mapping

At the end of season $y$, create the exposure-history distribution for season $y+1$.

Unvaccinated and uninfected:

$$
\rightarrow h=N.
$$

Vaccinated and uninfected:

$$
\rightarrow h=V.
$$

H1 infected and unvaccinated:

$$
E_{H1}^N+I_{H1}^N+R_{H1}^N
\rightarrow h=H1.
$$

H1 infected and vaccinated:

$$
E_{H1}^V+I_{H1}^V+R_{H1}^V
\rightarrow h=H1V.
$$

Similarly:

$$
H3 \rightarrow h=H3,
$$

$$
H3+V \rightarrow h=H3V,
$$

$$
B \rightarrow h=B,
$$

$$
B+V \rightarrow h=BV.
$$

At the start of the next influenza season, reset current-season vaccination status and retain only the previous-season exposure history.

---

## Algorithm 11. Seed the next influenza season

At the beginning of each season, introduce a small infectious seed for each subtype:

$$
I_{H1,0}>0,
\qquad
I_{H3,0}>0,
\qquad
I_{B,0}>0.
$$

For an initial reproduction, use:

$$
I_{0,m}=2.5\times10^{-6}.
$$

Repeat Algorithms 3-11 for each subsequent season.

---

## Algorithm 12. Fit the model after reproduction is validated

First reproduce plausible seasonal H1/H3/B dynamics with all parameters fixed.

Then release only:

$$
\beta_{H1},
\qquad
\beta_{H3},
\qquad
\beta_B.
$$

Keep fixed initially:

$$
a,
\qquad
\xi,
\qquad
\gamma_{1,m},
\qquad
\gamma_2,
\qquad
\alpha_{m,y}.
$$

Fit the three subtype-specific outputs jointly.

For example, define an objective function:

$$
L
=
L_{H1}
+
L_{H3}
+
L_B.
$$

The exact observation likelihood can later be chosen according to the surveillance data structure.

---

# Parameter Table

| Parameter | Meaning | Temporary value | Status |
|---|---|---:|---|
| $\beta_{H1}$ | H1N1 transmission rate | $0.3913\ \mathrm{day}^{-1}$ | Hill posterior median |
| $\beta_{H3}$ | H3N2 transmission rate | $0.3917\ \mathrm{day}^{-1}$ | Hill posterior median |
| $\beta_B$ | Combined influenza B transmission rate | $0.35815\ \mathrm{day}^{-1}$ | Mean of Hill BV/BY estimates |
| $\beta_{BV}$ | Hill B/Victoria transmission rate | $0.3510\ \mathrm{day}^{-1}$ | Hill source value |
| $\beta_{BY}$ | Hill B/Yamagata transmission rate | $0.3653\ \mathrm{day}^{-1}$ | Hill source value |
| $a$ | Residual susceptibility after same-strain infection last season | $0.7883$ | Hill estimate |
| $1-a$ | Implied same-strain protection | $0.2117$ | Derived |
| $\xi$ | Fraction of previous-season vaccine effectiveness retained | $0.0051$ | Hill estimate |
| $\gamma_{1,H1}$ | H1 latency-loss rate | $1/1.4=0.7143\ \mathrm{day}^{-1}$ | Fixed |
| $\gamma_{1,H3}$ | H3 latency-loss rate | $1/1.4=0.7143\ \mathrm{day}^{-1}$ | Fixed |
| $\gamma_{1,B}$ | B latency-loss rate | $1/0.6=1.6667\ \mathrm{day}^{-1}$ | Fixed |
| $\gamma_2$ | Recovery rate | $1/3.8=0.2632\ \mathrm{day}^{-1}$ | Fixed |
| $D$ | Natural mortality rate | $1/(81\times365)\ \mathrm{day}^{-1}$ | Optional |
| $\alpha_{H1}$ | Current-season H1 VE | $0.5525$ temporarily | Temporary |
| $\alpha_{H3}$ | Current-season H3 VE | $0.3045$ temporarily | Temporary |
| $\alpha_B$ | Current-season B VE | $0.5425$ temporarily | Derived simplification |
| $c_{m,y}$ | Residual susceptibility from last-season vaccination | $1-\xi\alpha_{m,y-1}$ | Computed |
| $I_{0,m}$ | Initial infectious fraction per subtype | $2.5\times10^{-6}$ | Fixed |
| $\epsilon_y$ | Ascertainment probability | $0.002$ initially | Placeholder |
| $\Delta t$ | Numerical integration time step | $1$ day | Fixed |
| $N$ | Normalized total population | $1$ | Fixed |

---

# Recommended Coding Strategy

Use one joint susceptible-history system rather than three independent full populations.

The same susceptible-history group is exposed to all three strain-specific forces of infection:

$$
\lambda_{H1},
\qquad
\lambda_{H3},
\qquad
\lambda_B.
$$

This prevents accidental population duplication.

Do not define separate full populations satisfying

$$
S_{H1}+E_{H1}+I_{H1}+R_{H1}=N,
$$

$$
S_{H3}+E_{H3}+I_{H3}+R_{H3}=N,
$$

and

$$
S_B+E_B+I_B+R_B=N.
$$

Instead, retain one joint population and let infection pathways be strain-specific.

---

# Recommended Development Sequence

1. Implement the fixed-parameter three-strain model.
2. Verify conservation of total population.
3. Verify that $f(h,m)$ only modifies susceptibility and is not recursively multiplied into $S$.
4. Verify current-season vaccination through $(1-\alpha_{m,y})$.
5. Verify end-of-season exposure-history remapping.
6. Reproduce plausible H1/H3/B seasonal epidemics.
7. Add observed subtype-specific data.
8. Estimate only $\beta_{H1}$, $\beta_{H3}$, and $\beta_B$.
9. Evaluate subtype-specific fit.
10. Add forecasting after the fitting implementation is stable.

