import { useMemo } from "react";
import { User, MapPin, Clock, Camera } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useEmployees } from "@/hooks/use-employees";
import { useAttendances } from "@/hooks/use-attendances";
import { getCurrentlyInOffice } from "@/lib/reporting";
import { format } from "date-fns";

export function LiveOccupancy() {
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
    <div className="glass-card rounded-2xl border border-border/40 overflow-hidden flex flex-col h-full">
      <div className="p-5 border-b border-border/40 bg-muted/20 flex items-center justify-between">
        <div>
          <h3 className="font-bold text-lg flex items-center gap-2">
             <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
             Live Occupancy
          </h3>
          <p className="text-xs text-muted-foreground">Who is currently in the building</p>
        </div>
        <Badge variant="secondary" className="px-3 py-1 font-mono">{inBuilding.length} PRESENT</Badge>
      </div>
      
      <ScrollArea className="flex-1">
        <div className="p-4 space-y-3">
          {inBuilding.length === 0 ? (
            <div className="py-12 text-center space-y-3 opacity-40">
               <User className="h-10 w-10 mx-auto text-muted-foreground" />
               <p className="text-sm">Building is currently empty</p>
            </div>
          ) : (
            inBuilding.map(({ employee, lastSeen }) => (
              <div key={employee.id} className="group p-3 rounded-xl border border-border/20 bg-muted/5 hover:bg-muted/20 transition-all duration-300 transform hover:scale-[1.02] cursor-default flex items-center gap-4">
                 <div className="h-12 w-12 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center relative shadow-inner overflow-hidden">
                    <User className="h-6 w-6 text-primary" />
                    <div className="absolute top-0 right-0 h-3 w-3 rounded-full bg-emerald-500 border-2 border-background shadow-sm" />
                 </div>
                 <div className="flex-1 min-w-0">
                    <p className="font-bold text-sm text-foreground truncate">{employee.name}</p>
                    <div className="flex items-center gap-3 mt-1 text-[10px] text-muted-foreground uppercase tracking-wider font-bold">
                       <span className="flex items-center gap-1.5"><Clock className="h-3 w-3" /> In at {lastSeen}</span>
                       <span className="flex items-center gap-1.5"><MapPin className="h-3 w-3" /> {employee.workLocation || "Main Office"}</span>
                    </div>
                 </div>
              </div>
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
