﻿# Автоматическая классификация обращений граждан

Авторство задачи: Рязанская область.

Оригинальное название: "Объективная оценка степени удовлетворенности населения"

## Краткое описание проблемы
Существует множество разрозненных источников поступления обратной связи от населения жалобы, предложения . Заявки поступают в различном формате в различные инстанции, которые, в свою очередь, не всегда компетентны по поступающему вопросу. Поступающая информация часто теряется, что ведет к нерешению проблем, росту однотипных повторяющихся заявок от населения и, как следствие, к повышению неудовлетворенности населения.

## Постановка задачи
Создать программу с машинным обучением, которая на основании содержания текста обращения могла бы определять тему обращения и ответственный орган власти.

## Данные
Представленный датасет содержит уже предразмеченные человеком обращения.
Входные данные содержатся в поле **text**.

## Ожидаемый результат
Разрабатываемый продукт должен генерировать:
- **theme** - генерируемый текст, содержащий ключевые факты из текст обращения
- **category** - предложение по отнесению обращения к определенной категории из справочника.
- **executor** - предполагаемый исполнитель, которому должно быть адресовано обращение.
В результате может быть создана система приема обращений граждан, которая автоматически направляет обращение непосредственному исполнителю.
