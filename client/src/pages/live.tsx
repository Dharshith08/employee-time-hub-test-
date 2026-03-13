import { useMemo } from "react";
import { User, MapPin, Clock, Shield, Activity, Users } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useEmployees } from "@/hooks/use-employees";
import { useAttendances } from "@/hooks/use-attendances";
import { getCurrentlyInOffice } from "@/lib/reporting";
import { format } from "date-fns";
import { motion, AnimatePresence } from "framer-motion";

export default function LivePage() {
  const { data: employees } = useEmployees();
  const { data: attendances } = useAttendances({
    dateFrom: format(new Date(), "yyyy-MM-dd"),
    dateTo: format(new Date(), "yyyy-MM-dd"),
  });

  const inBuilding = useMemo(() => {
    if (!employees || !attendances) return [];
    return getCurrentlyInOffice(employees, attendances);
  }, [employees, attendances]);

  return (
    <div className="p-6 md:p-8 space-y-8 animate-in fade-in duration-700">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-black text-foreground flex items-center gap-3">
             <Activity className="h-8 w-8 text-primary animate-pulse" />
             Live Command Center
          </h1>
          <p className="mt-1 text-muted-foreground uppercase text-[10px] font-black tracking-[0.2em]">
            Real-time occupancy and security monitoring
          </p>
        </div>
        
        <div className="flex items-center gap-4">
           <div className="glass-card px-4 py-2 border-border/40 flex items-center gap-3">
              <Users className="h-4 w-4 text-primary" />
              <div className="flex flex-col">
                 <span className="text-[10px] text-muted-foreground uppercase font-black">Total In</span>
                 <span className="text-lg font-black leading-none">{inBuilding.length}</span>
              </div>
           </div>
           <div className="glass-card px-4 py-2 border-border/40 flex items-center gap-3">
              <Shield className="h-4 w-4 text-emerald-500" />
              <div className="flex flex-col">
                 <span className="text-[10px] text-muted-foreground uppercase font-black">Security</span>
                 <span className="text-lg font-black leading-none text-emerald-500">ACTIVE</span>
              </div>
           </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <AnimatePresence mode="popLayout">
          {inBuilding.length === 0 ? (
            <div className="col-span-full py-32 text-center space-y-4 opacity-30">
               <User className="h-16 w-16 mx-auto text-muted-foreground" />
               <h2 className="text-xl font-bold uppercase tracking-widest">No Active Personnel Detected</h2>
            </div>
          ) : (
            inBuilding.map(({ employee, lastSeen }, index) => (
              <motion.div
                key={employee.id}
                initial={{ opacity: 0, y: 20, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.4, delay: index * 0.05 }}
                className="glass-card group relative p-6 border-border/40 hover:border-primary/40 transition-all duration-500 hover:shadow-[0_0_30px_rgba(var(--primary-rgb),0.1)] overflow-hidden"
              >
                {/* Background Accent */}
                <div className="absolute -right-4 -top-4 h-24 w-24 bg-primary/5 rounded-full blur-3xl group-hover:bg-primary/10 transition-colors" />
                
                <div className="relative z-10 space-y-6">
                  <div className="flex items-start justify-between">
                     <div className="h-16 w-16 rounded-2xl bg-gradient-to-br from-primary/20 to-primary/5 border border-primary/20 flex items-center justify-center shadow-inner relative">
                        <User className="h-8 w-8 text-primary" />
                        <div className="absolute -bottom-1 -right-1 h-5 w-5 rounded-full bg-emerald-500 border-4 border-background shadow-lg" />
                     </div>
                     <Badge variant="outline" className="bg-background/50 font-mono text-[10px] py-0">{employee.employeeCode}</Badge>
                  </div>

                  <div>
                    <h3 className="font-bold text-lg text-foreground truncate">{employee.name}</h3>
                    <p className="text-[10px] text-muted-foreground uppercase font-black tracking-tighter mt-1">{employee.department}</p>
                  </div>

                  <div className="pt-4 border-t border-border/20 space-y-3">
                     <div className="flex items-center gap-3 text-xs text-muted-foreground font-medium">
                        <Clock className="h-3.5 w-3.5 text-primary" />
                        <span>Entered at <span className="text-foreground font-bold">{lastSeen}</span></span>
                     </div>
                     <div className="flex items-center gap-3 text-xs text-muted-foreground font-medium">
                        <MapPin className="h-3.5 w-3.5 text-primary" />
                        <span className="truncate">{employee.workLocation || "Main Office"}</span>
                     </div>
                  </div>
                </div>

                <div className="absolute bottom-0 left-0 h-1 w-0 bg-primary group-hover:w-full transition-all duration-700" />
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
