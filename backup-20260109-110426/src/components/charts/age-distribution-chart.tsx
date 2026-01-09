'use client';

import { CartesianGrid, Line, LineChart, XAxis, YAxis, Tooltip } from 'recharts';
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from '@/components/ui/chart';
import { ageDistributionData, ageDistributionConfig } from '@/lib/data';

export default function AgeDistributionChart() {
  return (
    <ChartContainer config={ageDistributionConfig} className="h-[250px] w-full">
      <LineChart
        accessibilityLayer
        data={ageDistributionData}
        margin={{
          top: 20,
          right: 20,
          left: 10,
          bottom: 5,
        }}
      >
        <CartesianGrid vertical={false} />
        <XAxis
          dataKey="ageGroup"
          tickLine={false}
          axisLine={false}
          tickMargin={8}
        />
        <YAxis />
        <Tooltip content={<ChartTooltipContent />} />
        <Line
          dataKey="caries"
          type="monotone"
          stroke="var(--color-caries)"
          strokeWidth={2}
          dot={true}
        />
        <Line
          dataKey="gingivitis"
          type="monotone"
          stroke="var(--color-gingivitis)"
          strokeWidth={2}
          dot={true}
        />
      </LineChart>
    </ChartContainer>
  );
}
