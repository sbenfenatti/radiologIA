'use client';

import { Bar, BarChart, CartesianGrid, XAxis, YAxis, Tooltip } from 'recharts';
import {
  ChartContainer,
  ChartTooltipContent,
} from '@/components/ui/chart';
import { prevalenceChartData, prevalenceChartConfig } from '@/lib/data';

export default function PrevalenceChart() {
  return (
    <ChartContainer config={prevalenceChartConfig} className="h-[250px] w-full">
      <BarChart accessibilityLayer data={prevalenceChartData} margin={{ top: 20, right: 20, left: 0, bottom: 5 }}>
        <CartesianGrid vertical={false} />
        <XAxis
          dataKey="condition"
          tickLine={false}
          tickMargin={10}
          axisLine={false}
        />
        <YAxis />
        <Tooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
        <Bar dataKey="prevalence" radius={8} />
      </BarChart>
    </ChartContainer>
  );
}
