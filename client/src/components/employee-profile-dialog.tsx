import { useState, useMemo } from "react";
import { format, eachDayOfInterval, startOfMonth, endOfMonth, isSaturday, isSunday, subMonths, addMonths } from "date-fns";
import { 
  Calendar, 
  MapPin, 
  CalendarDays, 
  Clock, 
  TrendingUp, 
  ChevronLeft, 
  ChevronRight,
  Info,
  User,
  Zap
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useAttendances } from "@/hooks/use-attendances";
import { type Employee } from "@shared/schema";
import { PerformanceChart } from "./performance-chart";
import { calculatePerformanceScore } from "@/lib/reporting";

interface EmployeeProfileDialogProps {
  employee: Employee | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function EmployeeProfileDialog({ employee, open, onOpenChange }: EmployeeProfileDialogProps) {
  const [currentMonth, setCurrentMonth] = useState(new Date());
  
  const { data: attendances } = useAttendances(
    employee ? { 
      employeeId: employee.id,
      dateFrom: format(startOfMonth(subMonths(currentMonth, 1)), "yyyy-MM-dd"),
      dateTo: format(endOfMonth(addMonths(currentMonth, 1)), "yyyy-MM-dd"),
    } : { employeeId: -1 }
  );

  const monthDays = useMemo(() => {
    return eachDayOfInterval({
      start: startOfMonth(currentMonth),
      end: endOfMonth(currentMonth),
    });
  }, [currentMonth]);

  const stats = useMemo(() => {
    if (!attendances || !employee) return null;
    
    const relevant = attendances.filter(a => a.employeeId === employee.id);
    const totalHours = relevant.reduce((sum, a) => sum + (a.workingHours ?? 0), 0);
    const presentDays = new Set(relevant.filter(a => a.verificationStatus === "ENTRY" || a.verificationStatus === "EXIT").map(a => a.date)).size;
    
    return {
      totalHours,
      presentDays,
      avgHours: presentDays ? totalHours / presentDays : 0,
      score: calculatePerformanceScore(relevant),
    };
  }, [attendances, employee]);

  if (!employee) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh] flex flex-col glass-card border-border/40 p-0 overflow-hidden">
        <div className="p-6 bg-muted/20 border-b border-border/40 shrink-0">
          <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="h-16 w-16 rounded-full bg-primary/10 flex items-center justify-center text-primary border border-primary/20 shadow-inner shrink-0">
                <User className="h-8 w-8" />
              </div>
              <div>
                <DialogTitle className="text-2xl font-bold text-foreground">{employee.name}</DialogTitle>
                <div className="flex items-center gap-2 mt-1">
                   <Badge variant="outline" className="bg-background/50 font-mono tracking-tight">{employee.employeeCode}</Badge>
                   <span className="text-sm text-muted-foreground">{employee.department}</span>
                </div>
              </div>
            </div>
            <div className="flex flex-col md:items-end gap-1.5">
               <div className="flex items-center gap-2 text-sm text-muted-foreground">
                 <Calendar className="h-4 w-4" />
                 <span>Joined {employee.joiningDate ? format(new Date(employee.joiningDate), "PPP") : "N/A"}</span>
               </div>
               <div className="flex items-center gap-2 text-sm text-muted-foreground">
                 <MapPin className="h-4 w-4" />
                 <span>{employee.workLocation || "Main Office"}</span>
               </div>
            </div>
          </div>
        </div>

        <ScrollArea className="flex-1">
          <div className="p-6 space-y-6">
            {/* Stats Overview */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard label="Total Hours" value={`${stats?.totalHours.toFixed(1)}h`} icon={Clock} />
              <StatCard label="Avg. Daily" value={`${stats?.avgHours.toFixed(1)}h`} icon={TrendingUp} />
              <StatCard label="Present Days" value={String(stats?.presentDays || 0)} icon={CalendarDays} />
              <StatCard 
                label="Perf. Score" 
                value={String(stats?.score ?? 0)} 
                icon={Zap} 
                accent={(stats?.score ?? 0) > 80 ? "text-emerald-500" : (stats?.score ?? 0) > 50 ? "text-amber-500" : "text-rose-500"} 
              />
            </div>

            {/* AI Performance Insight */}
            <div className="glass-card rounded-2xl border border-border/40 p-6">
               <h4 className="font-semibold text-foreground flex items-center gap-2 mb-6">
                 <TrendingUp className="h-4 w-4 text-primary" />
                 Productivity Trend (Last 30 Days)
               </h4>
               <PerformanceChart records={attendances || []} />
            </div>

            {/* Attendance Calendar */}
            <div className="glass-card rounded-2xl border border-border/40 p-6">
               <div className="flex items-center justify-between mb-6">
                 <h4 className="font-semibold text-foreground flex items-center gap-2">
                   <CalendarDays className="h-4 w-4 text-primary" />
                   Monthly Attendance
                 </h4>
                 <div className="flex items-center gap-2 bg-muted/40 rounded-lg p-1 border border-border/40">
                   <button 
                     onClick={() => setCurrentMonth(subMonths(currentMonth, 1))} 
                     className="p-1 hover:bg-background rounded-md transition-all active:scale-95"
                   >
                     <ChevronLeft className="h-4 w-4" />
                   </button>
                   <span className="text-xs font-bold min-w-[120px] text-center uppercase tracking-wider">
                     {format(currentMonth, "MMMM yyyy")}
                   </span>
                   <button 
                     onClick={() => setCurrentMonth(addMonths(currentMonth, 1))} 
                     className="p-1 hover:bg-background rounded-md transition-all active:scale-95"
                   >
                     <ChevronRight className="h-4 w-4" />
                   </button>
                 </div>
               </div>

               <div className="grid grid-cols-7 gap-2">
                 {["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].map(day => (
                   <div key={day} className="text-center text-[10px] uppercase font-black text-muted-foreground/60 py-2 tracking-widest">{day}</div>
                 ))}
                 
                 {/* Empty slots for first week */}
                 {Array.from({ length: (startOfMonth(currentMonth).getDay() + 6) % 7 }).map((_, i) => (
                   <div key={`empty-${i}`} className="aspect-square opacity-20 bg-muted/5 rounded-xl border border-dashed border-border/20" />
                 ))}

                 {monthDays.map(day => {
                   const dateStr = format(day, "yyyy-MM-dd");
                   const record = attendances?.find(a => a.date === dateStr);
                   const isWeekend = isSaturday(day) || isSunday(day);
                   
                   let statusStyles = "bg-muted/5 border-border/40 hover:bg-muted/10";
                   let statusLabel = "No data";
                   
                   if (record) {
                     if (record.verificationStatus === "ENTRY" || record.verificationStatus === "EXIT") {
                       statusStyles = "bg-emerald-500/10 text-emerald-600 border-emerald-500/30 hover:bg-emerald-500/20";
                       statusLabel = "Present";
                     } else {
                        statusStyles = "bg-rose-500/10 text-rose-600 border-rose-500/30 hover:bg-rose-500/20";
                        statusLabel = "Failed verification";
                     }
                   } else if (!isWeekend && day < new Date()) {
                      statusStyles = "bg-rose-500/10 text-rose-600 border-rose-500/30 hover:bg-rose-500/20 opacity-70";
                      statusLabel = "Absent";
                   } else if (isWeekend) {
                      statusStyles = "bg-muted/10 opacity-30 border-dashed";
                      statusLabel = "Weekend";
                   }

                   return (
                     <div 
                       key={dateStr} 
                       className={`aspect-square rounded-xl border flex flex-col items-center justify-center gap-0.5 transition-all duration-300 transform hover:scale-105 cursor-help shadow-sm ${statusStyles}`}
                       title={statusLabel}
                     >
                       <span className="text-xs font-bold">{format(day, "d")}</span>
                       {record?.workingHours != null && record.workingHours > 0 && (
                         <span className="text-[8px] font-mono opacity-80">{record.workingHours}h</span>
                       )}
                     </div>
                   );
                 })}
               </div>

               <div className="flex flex-wrap items-center gap-4 mt-8 text-[10px] uppercase font-bold tracking-wider text-muted-foreground/60 border-t border-border/40 pt-6">
                 <div className="flex items-center gap-1.5"><div className="h-3 w-3 rounded bg-emerald-500/20 border border-emerald-500/40" /> Present</div>
                 <div className="flex items-center gap-1.5"><div className="h-3 w-3 rounded bg-rose-500/20 border border-rose-500/40" /> Absent / Failed</div>
                 <div className="flex items-center gap-1.5"><div className="h-3 w-3 rounded bg-muted/10 border border-border/40" /> Holiday / Weekend</div>
               </div>
            </div>
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}

function StatCard({ label, value, icon: Icon, accent }: { label: string; value: string; icon: any; accent?: string }) {
  return (
    <div className="glass-card rounded-2xl border border-border/30 p-4 flex flex-col gap-2 hover:translate-y-[-2px] transition-all duration-300">
      <div className="flex items-center justify-between">
        <div className="p-1.5 rounded-lg bg-primary/10 border border-primary/20">
          <Icon className="h-3.5 w-3.5 text-primary" />
        </div>
      </div>
      <div>
        <p className="text-[9px] uppercase font-black text-muted-foreground/70 tracking-[0.15em] leading-tight">{label}</p>
        <p className={`text-xl font-bold leading-none mt-1 ${accent || "text-foreground"}`}>{value}</p>
      </div>
    </div>
  );
}
