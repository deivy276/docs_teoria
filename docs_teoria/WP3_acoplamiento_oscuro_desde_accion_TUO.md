# WP3 — Derivación del acoplamiento oscuro desde la acción

## Estado
**Congelado como referencia EFT cosmológica de la TUO (IR).**

## Objetivo
Derivar de forma covariante y no fenomenológica el acoplamiento entre el tejido
(campo escalar efectivo `φ`) y la materia oscura fría (`CDM`), eliminando la activación
heurística tipo `tanh` y fijando la forma del intercambio de energía `Q` a partir
de la acción.

---

## 1. Decisión de marco

Se adopta el **marco de Einstein** como referencia EFT. La acción total se escribe como

\[
S_{\rm TUO}^{\rm eff}
=
\int d^4x\,\sqrt{-g}\,
\left[
\frac{M_{\rm Pl}^2}{2}R
-\frac12 Z(\phi)\,\nabla_\mu\phi \nabla^\mu \phi
-
V(\phi)
\right]
+
S_b[g_{\mu\nu},\psi_b]
+
S_r[g_{\mu\nu},\psi_r]
+
S_c[\tilde g^{(c)}_{\mu\nu},\psi_c].
\]

En la EFT de referencia se fija

\[
Z(\phi)=1,
\qquad
\tilde g^{(c)}_{\mu\nu}=C^2(\phi)\,g_{\mu\nu},
\qquad
C(\phi)=e^{\beta(\phi)}.
\]

La elección mínima de trabajo es

\[
\beta(\phi)=\beta_0\,\frac{\phi}{M_{\rm Pl}},
\qquad
\alpha(\phi)\equiv \frac{d\ln C}{d\phi}=\frac{\beta_0}{M_{\rm Pl}}.
\]

### Decisión de acoplamiento
- **Bariones**: acoplamiento mínimo a \(g_{\mu\nu}\).
- **Radiación**: acoplamiento mínimo a \(g_{\mu\nu}\).
- **CDM**: acoplamiento conforme a \(C^2(\phi) g_{\mu\nu}\).

Esto congela, a nivel EFT cosmológico, que la interacción del tejido es **exclusiva del sector oscuro**.

---

## 2. Definición de tensores energía–momento

Se definen:

\[
T^{(i)}_{\mu\nu}
=
-\frac{2}{\sqrt{-g}}\frac{\delta S_i}{\delta g^{\mu\nu}},
\qquad
T_{(i)}\equiv g^{\mu\nu}T^{(i)}_{\mu\nu}.
\]

Para el campo escalar del tejido,

\[
T^{(\phi)}_{\mu\nu}
=
\nabla_\mu\phi \nabla_\nu\phi
-
g_{\mu\nu}
\left[
\frac12 (\nabla\phi)^2 + V(\phi)
\right].
\]

La ecuación gravitacional es

\[
M_{\rm Pl}^2 G_{\mu\nu}
=
T^{(b)}_{\mu\nu}
+
T^{(r)}_{\mu\nu}
+
T^{(c)}_{\mu\nu}
+
T^{(\phi)}_{\mu\nu}.
\]

---

## 3. Variación respecto de \( \phi \)

La dependencia de \(S_c\) en \(\phi\) entra únicamente a través de la métrica efectiva
\(\tilde g^{(c)}_{\mu\nu}=C^2(\phi)g_{\mu\nu}\). Variando la acción se obtiene

\[
\delta S_c
=
\int d^4x\,\sqrt{-g}\,\alpha(\phi)\,T_{(c)}\,\delta\phi.
\]

Por tanto, la ecuación de movimiento del tejido queda:

\[
\Box\phi - V_{,\phi} = \alpha(\phi)\,T_{(c)}.
\]

Con la convención cosmológica usada en la TUO, para CDM no relativista

\[
T_{(c)} \simeq -\rho_c,
\]

y la ecuación se escribe en FLRW como

\[
\ddot\phi + 3H\dot\phi + V_{,\phi} = \alpha(\phi)\,\rho_c.
\]

---

## 4. No conservación individual del sector oscuro

Como solo el sector oscuro siente \(\tilde g^{(c)}_{\mu\nu}\), los sectores bariónico y radiativo
se conservan por separado:

\[
\nabla_\mu T^{\mu\nu}_{(b)} = 0,
\qquad
\nabla_\mu T^{\mu\nu}_{(r)} = 0.
\]

La materia oscura satisface:

\[
\nabla_\mu T^{\mu\nu}_{(c)}
=
-\alpha(\phi)\,T_{(c)}\,\nabla^\nu\phi.
\]

Esto garantiza conservación total:

\[
\nabla_\mu\left(
T^{\mu\nu}_{(b)}
+
T^{\mu\nu}_{(r)}
+
T^{\mu\nu}_{(c)}
+
T^{\mu\nu}_{(\phi)}
\right)=0.
\]

---

## 5. Forma cosmológica del intercambio \(Q\)

En un universo FLRW y para CDM con \(p_c=0\), la ecuación de continuidad oscura queda:

\[
\dot\rho_c + 3H\rho_c = -Q,
\]

con

\[
Q \equiv \alpha(\phi)\,\dot\phi\,\rho_c.
\]

Con la elección lineal de referencia,

\[
\alpha(\phi)=\frac{\beta_0}{M_{\rm Pl}},
\]

se obtiene

\[
Q = \frac{\beta_0}{M_{\rm Pl}}\,\dot\phi\,\rho_c.
\]

Si se usan las variables del código

\[
x \equiv \frac{\phi}{M_{\rm Pl}},
\qquad
N\equiv \ln a,
\qquad
y \equiv \frac{dx}{dN},
\qquad
\dot\phi = H M_{\rm Pl} y,
\]

entonces

\[
\boxed{
Q = \beta_0\,H\,y\,\rho_c
}
\]

que es exactamente la forma dinámica que la TUO debe usar en el sector de crecimiento y background.

**Resultado clave:**  
La activación del acoplamiento ya no depende de una función `tanh`. Depende únicamente
de la dinámica real del campo:

\[
Q \propto \beta_0\,y(N)\,\rho_c.
\]

---

## 6. Derivación covariante del bloqueo radiativo

La fuente del campo del tejido depende de la traza:

\[
T = -\rho + 3P.
\]

Para radiación,

\[
P_r = \rho_r/3
\quad\Longrightarrow\quad
T_r = -\rho_r + 3\rho_r/3 = 0.
\]

Luego, durante dominación radiativa, la radiación **no fuentea** directamente al campo \(\phi\).

Por tanto, en esa era:

\[
\ddot\phi + 3H\dot\phi + V_{,\phi} \simeq 0.
\]

Como además \(H\) es grande, la fricción de Hubble congela el campo y se obtiene naturalmente:

\[
\dot\phi \approx 0
\quad\Longrightarrow\quad
Q \approx 0.
\]

### Conclusión
El **bloqueo radiativo** no se introduce por fenomenología; emerge de forma covariante
a partir de la estructura de la acción y del hecho de que la traza de la radiación es nula.

---

## 7. Consecuencia física en eras cosmológicas

### Era de radiación
- \(T_r=0\)
- \(\dot\phi \approx 0\)
- \(Q \approx 0\)
- la TUO colapsa efectivamente a una cosmología casi-\(\Lambda\)CDM.

### Era de materia
- \(T_c \simeq -\rho_c \neq 0\)
- \(\phi\) comienza a rodar
- el acoplamiento oscuro se enciende dinámicamente.

### Era tardía
- \(Q\) puede modificar ligeramente:
  - \(H(z)\),
  - \(f\sigma_8(z)\),
  - \(S_8\),
  - la tasa efectiva de crecimiento.

---

## 8. Qué se congela oficialmente en WP3

### 8.1 Forma de la métrica oscura
\[
\tilde g^{(c)}_{\mu\nu}=e^{2\beta(\phi)}g_{\mu\nu}
\]

### 8.2 Forma mínima del acoplamiento
\[
\beta(\phi)=\beta_0\,\phi/M_{\rm Pl}
\]

### 8.3 Intercambio covariante
\[
Q=\alpha(\phi)\dot\phi \rho_c
\]

### 8.4 Forma en variables del código
\[
\boxed{Q=\beta_0 H y \rho_c}
\]

### 8.5 Mecanismo del bloqueo radiativo
\[
T_r=0 \Rightarrow \text{el campo no es fuenteado por radiación}
\]

---

## 9. Lo que WP3 NO cierra todavía

WP3 no resuelve aún:

1. la razón microfísica de por qué **solo** CDM acopla al tejido,
2. la posible generalización a una relación **disforme**,
3. la completion UV del acoplamiento,
4. el origen fundamental de la función \(\beta(\phi)\).

Eso queda para paquetes posteriores.

---

## 10. Resultado oficial de WP3

\[
\boxed{
\text{El acoplamiento oscuro de la TUO queda derivado covariantemente desde la acción,}
}
\]

\[
\boxed{
\text{con } Q=\beta_0 H y \rho_c \text{ como forma dinámica de referencia,}
}
\]

\[
\boxed{
\text{y con bloqueo radiativo natural derivado de } T_r=0.
}
\]
