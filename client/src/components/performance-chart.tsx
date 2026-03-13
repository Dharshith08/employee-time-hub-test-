import { useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { format, subDays, eachDayOfInterval } from "date-fns";
import { type AttendanceRecord } from "@/lib/reporting";

interface PerformanceChartProps {
  records: AttendanceRecord[];
  days?: number;
}

export function PerformanceChart({ records, days = 30 }: PerformanceChartProps) {
  const chartData = useMemo(() => {
    const end = new Date();
    const start = subDays(end, days - 1);
    const interval = eachDayOfInterval({ start, end });

    return interval.map((day) => {
      const dateStr = format(day, "yyyy-MM-dd");
      const record = records.find((r) => r.date === dateStr);
      
      return {
        date: format(day, "MMM d"),
        hours: record?.workingHours ?? 0,
        fullDate: dateStr,
      };
    });
  }, [records, days]);

  return (
    <div className="h-[200px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={chartData}
          margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
        >
          <defs>
            <linearGradient id="colorHours" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
              <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border) / 0.1)" />
          <XAxis 
            dataKey="date" 
            axisLine={false}
            tickLine={false}
            tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
            interval={Math.floor(days / 6)}
          />
          <YAxis 
            axisLine={false}
            tickLine={false}
            tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                return (
                  <div className="glass-card border-border/40 p-2 text-xs shadow-xl animate-in zoom-in-95 duration-200">
                    <p className="font-bold text-foreground mb-1">{payload[0].payload.date}</p>
                    <p className="text-primary font-mono">{payload[0].value} hours worked</p>
                  </div>
                );
              }
              return null;
            }}
          />
          <Area
            type="monotone"
            dataKey="hours"
            stroke="hsl(var(--primary))"
            strokeWidth={3}
            fillOpacity={1}
            fill="url(#colorHours)"
            animationDuration={1500}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
