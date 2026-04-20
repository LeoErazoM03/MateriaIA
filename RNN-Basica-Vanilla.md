# RNN Básica Vanilla

## Introducción

Una Vanilla RNN es un tipo de red neuronal recurrente que se caracteriza por utilizar la información del momento actual y también una parte de la información anterior para producir un nuevo estado. A diferencia de una red tradicional, este modelo tiene memoria a corto plazo, lo que le permite considerar eventos pasados al momento de calcular una salida.

En esta actividad, la Vanilla RNN se representa mediante la analogía de un “termómetro de emociones”, donde el estado emocional de cada día depende tanto del evento actual como de una fracción del estado emocional del día anterior. Esto permite observar cómo la red recuerda información previa, pero también cómo esa memoria se va desvaneciendo con el tiempo cuando no existen nuevos estímulos.

El objetivo de esta actividad es analizar el comportamiento básico de una Vanilla RNN a través de tres situaciones distintas: el desvanecimiento de un evento positivo fuerte, la recuperación frente a estados negativos acumulados y la comparación entre un estímulo único muy grande y varios estímulos pequeños pero constantes.

## Modelo matemático utilizado

Para resolver la actividad se usa la siguiente ecuación de una Vanilla RNN:

$$
h_t = x_t + 0.5h_{t-1}
$$

Donde:

- $h_t$: estado final del día actual  
- $x_t$: entrada o evento del día actual  
- $h_{t-1}$: estado del día anterior  
- $0.5$: fracción de memoria retenida del estado anterior  

Esto significa que cada día la red conserva únicamente el 50% del estado previo y lo combina con el evento nuevo. Debido a esto, la información antigua pierde fuerza progresivamente, lo que explica el problema del desvanecimiento de memoria en una Vanilla RNN.

## Misión 1: El Lunes Increíble (Demostrando el desvanecimiento)

### Planteamiento

En esta primera situación se analiza cómo un evento positivo muy fuerte pierde impacto con el paso de los días cuando no existen nuevos estímulos. La intención es demostrar que una Vanilla RNN no conserva de manera permanente la información pasada, sino que solo mantiene una fracción de ella en cada paso.

Los datos del problema son los siguientes:

- Día 1: $x_1 = 10$
- Día 2: $x_2 = 0$
- Día 3: $x_3 = 0$
- Día 4: $x_4 = 0$
- Día 5: $x_5 = 0$

Se toma como estado inicial del primer día:

$$
h_1 = 10
$$

### Desarrollo

Usando la ecuación:

$$
h_t = x_t + 0.5h_{t-1}
$$

se calcula el estado de cada día.

#### Día 2

$$
h_2 = x_2 + 0.5h_1
$$

$$
h_2 = 0 + 0.5(10)
$$

$$
h_2 = 5
$$

#### Día 3

$$
h_3 = x_3 + 0.5h_2
$$

$$
h_3 = 0 + 0.5(5)
$$

$$
h_3 = 2.5
$$

#### Día 4

$$
h_4 = x_4 + 0.5h_3
$$

$$
h_4 = 0 + 0.5(2.5)
$$

$$
h_4 = 1.25
$$

#### Día 5

$$
h_5 = x_5 + 0.5h_4
$$

$$
h_5 = 0 + 0.5(1.25)
$$

$$
h_5 = 0.625
$$

### Resultado

El estado final del viernes es:

$$
\boxed{h_5 = 0.625}
$$

### Interpretación

Aunque el lunes ocurrió un evento muy positivo con valor de $10$, al pasar los días ese efecto fue disminuyendo hasta llegar a $0.625$ el viernes. Esto demuestra que la Vanilla RNN va perdiendo la memoria de eventos antiguos debido a que solo conserva una parte del estado anterior en cada paso. En otras palabras, un estímulo fuerte aislado no se mantiene por mucho tiempo si no existen nuevas entradas que lo refuercen.

## Misión 2: El Rescate Emocional (Superando el pasado)

### Planteamiento

En esta segunda situación se busca determinar qué tan grande debe ser un evento positivo para lograr que el estado final del día 4 sea mayor que cero, después de haber acumulado varios eventos negativos en los días anteriores.

Los datos del problema son:

- Día 1: $x_1 = -6$
- Día 2: $x_2 = -4$
- Día 3: $x_3 = 0$
- Día 4: $x_4 = ?$

Se considera como estado inicial:

$$
h_0 = 0
$$

### Desarrollo

Se utiliza nuevamente la ecuación de la Vanilla RNN:

$$
h_t = x_t + 0.5h_{t-1}
$$

#### Día 1

$$
h_1 = x_1 + 0.5h_0
$$

$$
h_1 = -6 + 0.5(0)
$$

$$
h_1 = -6
$$

#### Día 2

$$
h_2 = x_2 + 0.5h_1
$$

$$
h_2 = -4 + 0.5(-6)
$$

$$
h_2 = -4 - 3
$$

$$
h_2 = -7
$$

#### Día 3

$$
h_3 = x_3 + 0.5h_2
$$

$$
h_3 = 0 + 0.5(-7)
$$

$$
h_3 = -3.5
$$

#### Día 4

Ahora se desea que el estado final del día 4 sea positivo:

$$
h_4 = x_4 + 0.5h_3
$$

Sustituyendo el valor de $h_3$:

$$
h_4 = x_4 + 0.5(-3.5)
$$

$$
h_4 = x_4 - 1.75
$$

Para que el estado final sea positivo, debe cumplirse que:

$$
h_4 > 0
$$

Entonces:

$$
x_4 - 1.75 > 0
$$

$$
x_4 > 1.75
$$

### Resultado

El evento del día 4 debe ser mayor que:

$$
\boxed{x_4 > 1.75}
$$

Si se desea un valor entero mínimo, entonces el evento debe ser:

$$
\boxed{x_4 = 2}
$$

### Interpretación

Esta misión muestra que, cuando la red ha acumulado estados negativos en días anteriores, no basta con un evento levemente positivo para revertir la situación. Se necesita una entrada nueva con suficiente magnitud para compensar la memoria negativa arrastrada por la red. Esto demuestra cómo el estado pasado sigue influyendo en el presente, incluso cuando el evento actual es diferente.

## Misión 3: Constancia vs. El Pico (Cómo aprende la red)

### Planteamiento

En esta tercera situación se comparan dos escenarios distintos para analizar cuál produce un mayor efecto al final del día 5. El propósito es observar si una Vanilla RNN recuerda mejor un solo evento grande al inicio o varios eventos pequeños pero constantes a lo largo de varios días.

Los escenarios son los siguientes:

**Escenario A:**

- Día 1: $x_1 = 10$
- Día 2: $x_2 = 0$
- Día 3: $x_3 = 0$
- Día 4: $x_4 = 0$
- Día 5: $x_5 = 0$

**Escenario B:**

- Día 1: $x_1 = 3$
- Día 2: $x_2 = 3$
- Día 3: $x_3 = 3$
- Día 4: $x_4 = 3$
- Día 5: $x_5 = 3$

Se utiliza el mismo modelo:

$$
h_t = x_t + 0.5h_{t-1}
$$

### Desarrollo

### Escenario A: Un pico aislado

Este caso ya fue calculado en la Misión 1.

$$
h_1 = 10
$$

$$
h_2 = 0 + 0.5(10) = 5
$$

$$
h_3 = 0 + 0.5(5) = 2.5
$$

$$
h_4 = 0 + 0.5(2.5) = 1.25
$$

$$
h_5 = 0 + 0.5(1.25) = 0.625
$$

Por lo tanto, el estado final del día 5 en el Escenario A es:

$$
\boxed{h_5 = 0.625}
$$

### Escenario B: Pequeñas alegrías constantes

Se toma como estado inicial:

$$
h_0 = 0
$$

#### Día 1

$$
h_1 = x_1 + 0.5h_0
$$

$$
h_1 = 3 + 0.5(0)
$$

$$
h_1 = 3
$$

#### Día 2

$$
h_2 = x_2 + 0.5h_1
$$

$$
h_2 = 3 + 0.5(3)
$$

$$
h_2 = 3 + 1.5
$$

$$
h_2 = 4.5
$$

#### Día 3

$$
h_3 = x_3 + 0.5h_2
$$

$$
h_3 = 3 + 0.5(4.5)
$$

$$
h_3 = 3 + 2.25
$$

$$
h_3 = 5.25
$$

#### Día 4

$$
h_4 = x_4 + 0.5h_3
$$

$$
h_4 = 3 + 0.5(5.25)
$$

$$
h_4 = 3 + 2.625
$$

$$
h_4 = 5.625
$$

#### Día 5

$$
h_5 = x_5 + 0.5h_4
$$

$$
h_5 = 3 + 0.5(5.625)
$$

$$
h_5 = 3 + 2.8125
$$

$$
h_5 = 5.8125
$$

Por lo tanto, el estado final del día 5 en el Escenario B es:

$$
\boxed{h_5 = 5.8125}
$$

### Comparación de resultados

- **Escenario A:**

$$
\boxed{h_5 = 0.625}
$$

- **Escenario B:**

$$
\boxed{h_5 = 5.8125}
$$

Se observa que el Escenario B produce un estado final mucho mayor que el Escenario A.

### Interpretación

Esta comparación demuestra que una Vanilla RNN recuerda mejor la información reciente y constante que un solo evento muy grande ocurrido al inicio. Aunque el pico de $10$ del Escenario A parece más fuerte al principio, su efecto se va desvaneciendo rápidamente. En cambio, en el Escenario B, las entradas positivas continuas refuerzan el estado en cada paso, permitiendo que el valor final sea mucho más alto.

Esto confirma que, en una Vanilla RNN, la constancia tiene mayor impacto que un estímulo aislado cuando se evalúa el efecto a lo largo del tiempo.

## Conclusión general

A partir del desarrollo de las tres misiones, se puede concluir que una Vanilla RNN combina la información actual con una parte del estado anterior, lo que le permite tener memoria de corto plazo. Sin embargo, esa memoria no se conserva completamente, sino que se va reduciendo en cada paso debido al factor de retención aplicado al estado previo.

En la primera misión se comprobó que un evento positivo muy fuerte puede perder casi todo su efecto después de varios días si no existen nuevas entradas que lo refuercen. En la segunda misión se observó que, cuando se acumulan estados negativos, se necesita un nuevo evento de suficiente magnitud para revertir esa tendencia y lograr un estado positivo. Finalmente, en la tercera misión se demostró que una secuencia de entradas pequeñas pero constantes produce un efecto final mayor que un solo evento grande al inicio, ya que la red da más peso a la información reciente y repetida.

En conclusión, esta actividad permite entender de manera sencilla el funcionamiento básico de una Vanilla RNN y también su principal limitación: el desvanecimiento de la memoria con el tiempo. Por ello, aunque este tipo de red puede modelar secuencias, no resulta tan eficiente cuando se necesita conservar información importante durante muchos pasos.
