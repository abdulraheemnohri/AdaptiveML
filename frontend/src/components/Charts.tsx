import { LineChart as RechartsLineChart, Line, BarChart as RechartsBarChart, Bar, PieChart as RechartsPieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts'

const COLORS = [
  'hsl(var(--primary))',
  'hsl(var(--secondary))',
  '#3b82f6',
  '#10b981',
  '#f59e0b',
  '#ef4444',
  '#8b5cf6',
  '#06b6d4',
]

interface LineChartProps {
  data: { name: string; value: number }[]
  className?: string
}

export function LineChart({ data, className }: LineChartProps) {
  return (
    <div className={className} style={{ width: '100%', height: 300 }}>
      <ResponsiveContainer width="100%" height="100%">
        <RechartsLineChart data={data}>
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--background))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '6px',
            }}
          />
          <Line type="monotone" dataKey="value" stroke={COLORS[0]} strokeWidth={2} />
        </RechartsLineChart>
      </ResponsiveContainer>
    </div>
  )
}

interface BarChartProps {
  data: { name: string; value: number }[]
  className?: string
}

export function BarChart({ data, className }: BarChartProps) {
  return (
    <div className={className} style={{ width: '100%', height: 300 }}>
      <ResponsiveContainer width="100%" height="100%">
        <RechartsBarChart data={data}>
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--background))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '6px',
            }}
          />
          <Bar dataKey="value" fill={COLORS[0]} />
        </RechartsBarChart>
      </ResponsiveContainer>
    </div>
  )
}

interface PieChartProps {
  data: { name: string; value: number }[]
  className?: string
}

export function PieChart({ data, className }: PieChartProps) {
  return (
    <div className={className} style={{ width: '100%', height: 300 }}>
      <ResponsiveContainer width="100%" height="100%">
        <RechartsPieChart>
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--background))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '6px',
            }}
          />
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
        </RechartsPieChart>
      </ResponsiveContainer>
    </div>
  )
}