# WP1 — Cierre cinemático del contenido de grados de libertad de la TUO

## Estado
Aprobado como **hipótesis cinemática de referencia** para la TUO en el régimen cosmológico infrarrojo (IR).

## Objetivo
Fijar de manera explícita qué grados de libertad se consideran **físicos** en la formulación cosmológica actual de la TUO, y qué estructura tensorial se interpreta como:

1. física y propagante,
2. auxiliar o gauge,
3. o pendiente de demostración formal en WP2.

WP1 **no** demuestra todavía ausencia de fantasmas. Ese cierre pertenece a WP2. Aquí solo se congela el **contenido cinemático de trabajo**.

---

## 1. Principio de trabajo adoptado
La TUO se congela, a nivel cosmológico efectivo, como una teoría **escalar–tensorial infrarroja** cuyo contenido físico es:

- 2 grados de libertad tensoriales de la métrica gravitacional estándar,
- 1 grado de libertad escalar adicional del tejido, `phi`.

En notación compacta:

```math
\text{d.o.f. físicos de la TUO en IR} = 2_{\rm GR} + 1_{\phi}.
```

Esto significa que **la versión operativa actual de la TUO no se interpreta como una teoría cosmológica de espín-2 masivo pleno**. El tensor `A_{mu nu}` se mantiene como objeto geométrico “padre” o estructura subyacente, pero no como campo masivo completo propagando 5 grados de libertad en el sector cosmológico usado hasta ahora.

---

## 2. Decisión de ruta
### Ruta congelada en WP1
Se adopta la siguiente hipótesis de trabajo:

> La completion infrarroja físicamente relevante de la TUO es una **teoría escalar–tensorial** donde el único grado de libertad extra ligero y propagante del tejido es `phi`.

### Ruta no adoptada en WP1
No se adopta, por ahora, una completion tipo espín-2 masivo no lineal completa (por ejemplo, tipo dRGT/Hassan–Rosen), por las razones siguientes:

- no es el contenido de grados de libertad que usa el simulador cosmológico actual,
- introduciría una fenomenología tensorial adicional no exigida por los ajustes actuales,
- obligaría a confrontar inmediatamente el sector de ondas gravitacionales y modos masivos del gravitón,
- y desviaría el programa de trabajo del objetivo cosmológico efectivo inmediato.

**Conclusión:** la TUO sigue su desarrollo como **EFT escalar–tensorial cosmológica**.

---

## 3. Campo geométrico “padre” del tejido
Se mantiene la idea conceptual de que existe una estructura tensorial subyacente `A_{mu nu}`. Sin embargo, su papel cinemático se congela así:

```math
A_{\mu\nu} = \frac14 \phi\, g_{\mu\nu} + \hat A_{\mu\nu},
\qquad \hat A^\mu{}_{\mu}=0.
```

Interpretación:

- `phi` = traza/modo escalar efectivo del tejido,
- `hat A_{mu nu}` = sector trazaless del tensor padre.

### Hipótesis cinemática congelada
En el régimen IR cosmológico:

- `phi` es el **único modo extra ligero y propagante** del tejido,
- `hat A_{mu nu}` se trata como:
  - auxiliar,
  - puro gauge,
  - o suficientemente pesado como para no propagarse en el régimen cosmológico considerado.

**Importante:** esta afirmación es todavía una hipótesis de trabajo y deberá justificarse con análisis de restricciones en WP2.

---

## 4. Simetría cinemática de referencia
La motivación geométrica de la TUO en WP1 se fija como **TDiff/WTDiff-motivada** o equivalente escalar–tensorial, en el siguiente sentido:

- la teoría debe poseer una simetría o estructura de restricciones capaz de eliminar modos no físicos del sector tensorial,
- el volumen/traza deja de ser un artefacto puro de coordenadas y puede emerger como grado de libertad escalar físico,
- la formulación cosmológica efectiva debe ser equivalente, en IR, a una teoría con un solo escalar extra.

### Formulación operativa congelada
WP1 **no** congela todavía una acción tensorial fundamental única con TDiff completa, pero sí congela el siguiente criterio:

> La completion futura del tensor `A_{mu nu}` deberá respetar una estructura de simetría y restricciones compatible con que solo sobreviva un modo escalar físico en IR.

---

## 5. Conteo cinemático de grados de libertad
### 5.1 Métrica gravitacional
La métrica `g_{mu nu}` mantiene el contenido estándar de GR:

- 2 grados de libertad tensoriales físicos.

### 5.2 Campo del tejido
El tejido contribuye, a nivel efectivo, con:

- 1 grado de libertad escalar físico: `phi`.

### 5.3 Sectores no propagantes en IR
En la EFT congelada se asume que no sobreviven en el régimen cosmológico:

- modos vectoriales propagantes del tejido,
- modos tensoriales masivos adicionales del sector `A_{mu nu}`.

### Conteo total congelado
```math
N_{\rm dof}^{\rm TUO,IR} = 2 + 1 = 3.
```

---

## 6. Acción efectiva coherente con WP1
La acción efectiva cosmológica ya congelada en WP0 queda reinterpretada, en WP1, como la acción del sector físico IR:

```math
S_{\rm TUO}^{\rm eff}
=
\int d^4x\,\sqrt{-g}
\left[
\frac{M_{\rm Pl}^2}{2}R
-\frac12 (\nabla\phi)^2
-V(\phi)
\right]
+S_b[g,\psi_b]
+S_r[g,\psi_r]
+S_c[e^{2\beta(\phi)}g,\psi_c].
```

Esto implica que WP1 congela la interpretación siguiente:

- esta acción **no es solo un ansatz fenomenológico**,
- sino la **acción IR física** de la TUO mientras no se cierre la estructura tensorial profunda.

---

## 7. Estructura de acoplamiento material congelada
### Bariones y radiación
Se acoplan mínimamente a `g_{mu nu}`.

### Materia oscura fría
Se acopla a una métrica efectiva conforme:

```math
\tilde g^{(c)}_{\mu\nu} = e^{2\beta(\phi)} g_{\mu\nu}.
```

En la aproximación mínima:

```math
\beta(\phi)=\beta_0\phi/M_{\rm Pl}.
```

### Consecuencia cinemática
El único canal nuevo de interacción relevante en cosmología lineal se organiza a través del campo escalar `phi`.

---

## 8. Implicaciones para el sector lineal de crecimiento
WP1 congela también la interpretación del sector de crecimiento como una modificación inducida por un único escalar efectivo:

```math
Q \propto \beta_0\,\phi'(N)\,\rho_c,
```

```math
\mu(k,a)=1+\frac{2\beta_0^2k^2}{k^2+a^2m_{\rm eff}^2(a)}.
```

Esto se entiende como la manifestación IR de:

- una interacción oscuro–tejido gobernada por `phi`,
- una gravedad efectiva dependiente de escala mediada por el modo escalar del tejido.

---

## 9. Lo que WP1 cierra y lo que NO cierra
### WP1 cierra
- cuál es el contenido físico de la EFT cosmológica,
- que el único modo extra propagante es `phi`,
- que la TUO actual debe leerse como teoría escalar–tensorial IR,
- y que el tensor `A_{mu nu}` queda relegado a estructura subyacente no propagante en el régimen cosmológico probado.

### WP1 NO cierra
- la formulación tensorial UV/parent exacta,
- la demostración ghost-free,
- el análisis Hamiltoniano,
- la prueba de ausencia de modos patológicos en `hat A_{mu nu}`,
- ni la completion de espín-2 masivo.

Eso se transfiere explícitamente a WP2.

---

## 10. Criterios de éxito de WP1
WP1 se considera satisfecho si quedan fijados inequívocamente:

1. el número de grados de libertad físicos en IR,
2. la interpretación del campo `phi` como único modo extra ligero,
3. el estatuto no propagante de `hat A_{mu nu}` en el régimen cosmológico actual,
4. y la ruta de continuidad con WP2.

Estos cuatro criterios se consideran cumplidos por la presente congelación.

---

## 11. Punto de paso a WP2
WP2 deberá responder a la pregunta:

> ¿Puede esta estructura cinemática cerrarse como teoría sin fantasmas y con Hamiltoniano bien definido?

Para ello, WP2 deberá demostrar explícitamente una de estas dos cosas:

### Opción A
La formulación tensorial subyacente posee restricciones suficientes para eliminar todos los modos patológicos y dejar solo `phi` en IR.

### Opción B
La TUO se declara definitivamente como teoría escalar–tensorial fundamental en su régimen cosmológico, abandonando cualquier pretensión de un espín-2 extra propagante.

---

## 12. Conclusión formal de WP1
La TUO queda congelada, en su régimen cosmológico operativo, como una **teoría escalar–tensorial infrarroja** con:

- 2 grados de libertad tensoriales de GR,
- 1 grado de libertad escalar físico del tejido,
- acoplamiento oscuro conforme,
- y un tensor padre `A_{mu nu}` cuya traza define el modo efectivo `phi`, mientras el sector trazaless no se considera propagante en IR.

En una frase:

> **WP1 fija que la TUO cosmológica actual no es una teoría de espín-2 masivo, sino una EFT escalar–tensorial con origen tensorial subyacente aún pendiente de cierre en WP2.**
