import { useEffect, useRef } from 'react'
import * as echarts from 'echarts/core'
import { LineChart, BarChart, ScatterChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsCoreOption } from 'echarts/core'

echarts.use([LineChart, BarChart, ScatterChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer])
export default function Chart({ option, label }: { option: EChartsCoreOption; label: string }) {
  const element = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!element.current) return
    const chart = echarts.init(element.current)
    chart.setOption({ color: ['#197f77', '#d69a42', '#5f7b9a'], textStyle: { fontFamily: 'system-ui, sans-serif', color: '#5d6c73' }, ...option })
    const resize = new ResizeObserver(() => chart.resize())
    resize.observe(element.current)
    return () => { resize.disconnect(); chart.dispose() }
  }, [option])
  return <div ref={element} className="chart" role="img" aria-label={label} />
}
