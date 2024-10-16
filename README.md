Fjernede Lande
Lande med over 1000 manglende værdier blev fjernet fra datasættet. I alt blev 93 lande fjernet. Nogle af disse lande inkluderer:
	• Afghanistan, Albania, Angola, Antigua and Barbuda, Armenia
	• Bosnia and Herzegovina, Botswana, Burkina Faso, Cambodia, Chad
	• Cuba, Dominican Republic, El Salvador, Ethiopia, Fiji
	• French Guiana, Gabon, Georgia, Ghana, Guatemala
	• Honduras, Jamaica, Kenya, Lesotho, Liberia
	• Mozambique, Namibia, Niger, Nigeria, Papua New Guinea
	• Sierra Leone, Somalia, South Sudan, Sudan, Tuvalu, Yemen

Fjernede Kolonner
Kolonner med over 1000 manglende værdier blev også fjernet. Kun én kolonne opfyldte denne betingelse:
	• Financial flows to developing countries (US $)
Denne kolonne havde for mange manglende værdier på tværs af alle lande og år og blev fjernet for at sikre datakvaliteten.


Sammenfatning af Datasættets Kvalitet og Struktur
1. Generel Information
	• Antal rækker (observationer): 1,512
	• Antal kolonner (variable): 148
	• Antal unikke lande: 72
	• Tidsinterval: 2000 til 2020 (21 år)
	• Samlet antal manglende værdier: 15,900
	• Antal kolonner med manglende værdier: 79
	• Gennemsnitlige manglende værdier pr. kolonne: 201.27
Dette betyder, at datasættet dækker 72 lande over en periode på 21 år. Der er dog en betydelig mængde manglende værdier, hvilket tyder på, at visse variabler er mere komplette end andre.
2. Kolonnenavne og Datatyper
	• Datasættet indeholder primært numeriske kolonner (float64) og kun én tekstkolonne (country) samt én heltalskolonne (year).
	• Kolonnenavne og datatyper er generelt konsistente og passer til numeriske analyser og maskinlæring.
3. Statistik over Numeriske Kolonner
	• De numeriske kolonner har en varieret fordeling med gennemsnit, standardafvigelser, minimum- og maksimumværdier, hvilket indikerer en bred dækning af variabler.
	• Access to electricity (% of population) og Access to clean fuels for cooking har høj gennemsnitlig dækning, hvilket viser, at mange lande har næsten fuld adgang til elektricitet og rene brændstoffer.
	• Kolonner som wind_energy_per_capita og Electricity from fossil fuels (TWh) viser stor varians og tilstedeværelsen af outliers, hvilket kan indikere forskelle i energiproduktionen mellem lande.
4. Høj Variabilitet
	• Kolonner med den højeste standardafvigelse (variabilitet) omfatter økonomiske og demografiske variable som:
		○ gdp (bruttonationalprodukt)
		○ population (befolkning)
		○ Land Area(Km2) (landareal)
		○ Value_co2_emissions_kt_by_country (CO₂-udledninger)
		○ oil_prod_per_capita (olieproduktion pr. indbygger)
Disse kolonner har stor variabilitet, hvilket tyder på markante forskelle mellem landene og muligvis nogle ekstremværdier.
5. Potentielle Outliers
	• Flere kolonner viser høje maksimumværdier, der ligger langt over 75%-kvartilen. Eksempler omfatter:
		○ Electricity from fossil fuels (TWh), hvor maksimumværdi (5,184.13) langt overstiger 75%-kvartilen.
		○ wind_energy_per_capita, hvor maksimumværdi (7,361.917) indikerer en signifikant forskel i energiproduktion i visse lande.
		○ Renewable-electricity-generating-capacity-per-capita og andre energirelaterede kolonner, der viser stor forskel i værdier mellem lande og potentielt outliers.


Kolonnerne i kategorier med en kort forklaring på hver kategori og de tilhørende kolonner:
1. Geografisk og Demografisk Information
Disse kolonner indeholder information om landets navn, placering, befolkningstal og landareal.
	• country: Landets navn.
	• year: Året for dataindsamling.
	• iso_code: Landets ISO-kode.
	• population: Befolkningstal.
	• Density (P/Km2): Befolkningstæthed pr. kvadratkilometer.
	• Land Area(Km2): Landareal i kvadratkilometer.
	• Latitude: Geografisk breddegrad.
	• Longitude: Geografisk længdegrad.
2. Økonomisk Information
Disse kolonner beskriver økonomiske indikatorer som BNP og vækst, der giver indblik i landets økonomiske situation.
	• gdp: Bruttonationalprodukt (BNP) i dollars.
	• gdp_growth: Vækstrate for BNP.
	• gdp_per_capita: BNP pr. indbygger, som viser velstandsniveauet i landet.
3. Elforbrug og Produktionen af El
Disse kolonner beskriver elforbrug og elproduktion i landet, som ofte er vigtige mål for energiforbrug.
	• electricity_demand: Samlet elforbrug i landet.
	• electricity_demand_per_capita: Elforbrug pr. indbygger.
	• electricity_generation: Samlet elproduktion i landet.
	• electricity_share_energy: Elproduktion som andel af den samlede energiforsyning.
4. Kilder til Elproduktion
Disse kolonner viser elproduktionen fra forskellige energikilder, hvilket kan give et indtryk af energimixet i landet.
	• Electricity from fossil fuels (TWh): Elproduktion fra fossile brændstoffer.
	• Electricity from nuclear (TWh): Elproduktion fra kernekraft.
	• Electricity from renewables (TWh): Elproduktion fra vedvarende energikilder.
	• Low-carbon electricity (% electricity): Andel af lav-karbon elektricitet som en procentdel af den samlede elproduktion.
	• biofuel_electricity: Elproduktion fra biobrændstof.
	• coal_electricity: Elproduktion fra kul.
	• fossil_electricity: Elproduktion fra fossile brændstoffer.
	• gas_electricity: Elproduktion fra naturgas.
	• hydro_electricity: Elproduktion fra vandkraft.
	• nuclear_electricity: Elproduktion fra kernekraft.
	• oil_electricity: Elproduktion fra olie.
	• renewables_electricity: Elproduktion fra vedvarende kilder.
	• solar_electricity: Elproduktion fra solenergi.
	• wind_electricity: Elproduktion fra vindenergi.
5. Forbrug af Energiressourcer
Denne kategori beskriver forbruget af forskellige energikilder, herunder fossile brændstoffer og vedvarende energikilder.
	• primary_energy_consumption: Samlet energiforbrug i landet.
	• biofuel_consumption: Forbrug af biobrændstof.
	• coal_consumption: Forbrug af kul.
	• fossil_fuel_consumption: Forbrug af fossile brændstoffer.
	• gas_consumption: Forbrug af naturgas.
	• hydro_consumption: Forbrug af vandkraft.
	• nuclear_consumption: Forbrug af kernekraft.
	• oil_consumption: Forbrug af olie.
	• renewables_consumption: Forbrug af vedvarende energi.
	• solar_consumption: Forbrug af solenergi.
	• wind_consumption: Forbrug af vindenergi.
	• other_renewable_consumption: Forbrug af andre vedvarende energikilder.
6. Ændring i Energiforbrug over Tid
Disse kolonner beskriver årlige ændringer i forbruget af forskellige energikilder, som kan hjælpe med at forstå udviklingen over tid.
	• biofuel_cons_change_pct: Ændring i biobrændstof forbrug (procent).
	• coal_cons_change_pct: Ændring i kulforbrug (procent).
	• fossil_cons_change_pct: Ændring i fossilt brændstofforbrug (procent).
	• gas_cons_change_pct: Ændring i gasforbrug (procent).
	• hydro_cons_change_pct: Ændring i vandkraftforbrug (procent).
	• nuclear_cons_change_pct: Ændring i kernekraftsforbrug (procent).
	• oil_cons_change_pct: Ændring i oliefrobrug (procent).
	• renewables_cons_change_pct: Ændring i vedvarende energiforbrug (procent).
	• solar_cons_change_pct: Ændring i solenergiforbrug (procent).
	• wind_cons_change_pct: Ændring i vindenergiforbrug (procent).
7. Energiproduktion og Forbrug pr. Indbygger
Disse kolonner viser produktionen og forbruget af forskellige energikilder pr. indbygger.
	• biofuel_elec_per_capita: Elproduktion fra biobrændstof pr. indbygger.
	• coal_elec_per_capita: Elproduktion fra kul pr. indbygger.
	• fossil_elec_per_capita: Elproduktion fra fossile brændstoffer pr. indbygger.
	• gas_elec_per_capita: Elproduktion fra naturgas pr. indbygger.
	• hydro_elec_per_capita: Elproduktion fra vandkraft pr. indbygger.
	• nuclear_elec_per_capita: Elproduktion fra kernekraft pr. indbygger.
	• oil_elec_per_capita: Elproduktion fra olie pr. indbygger.
	• renewables_elec_per_capita: Elproduktion fra vedvarende energikilder pr. indbygger.
	• solar_elec_per_capita: Elproduktion fra solenergi pr. indbygger.
	• wind_elec_per_capita: Elproduktion fra vindkraft pr. indbygger.
8. Andel af Energiforbrug efter Kilde
Disse kolonner beskriver fordelingen af elproduktion fra forskellige kilder som procentdele af den samlede elproduktion.
	• biofuel_share_elec: Biobrændstofs andel af elproduktion.
	• coal_share_elec: Kuls andel af elproduktion.
	• fossil_share_elec: Fossile brændstoffers andel af elproduktion.
	• gas_share_elec: Naturgas’ andel af elproduktion.
	• hydro_share_elec: Vandkrafts andel af elproduktion.
	• nuclear_share_elec: Kernekrafts andel af elproduktion.
	• oil_share_elec: Olies andel af elproduktion.
	• renewables_share_elec: Vedvarende energis andel af elproduktion.
	• solar_share_elec: Solenergis andel af elproduktion.
	• wind_share_elec: Vindkrafts andel af elproduktion.
9. Drivhusgasser og Emissioner
Disse kolonner beskriver miljøpåvirkningen fra energiforbruget, herunder CO₂-emissioner og andre drivhusgasser.
	• carbon_intensity_elec: CO₂-intensitet af elektricitet, målt i forhold til elproduktionen.
	• greenhouse_gas_emissions: Samlet udledning af drivhusgasser.
	• Value_co2_emissions_kt_by_country: CO₂-udledninger målt i tusinder af tons.
10. Anden Energiinformation
Andre energirelaterede mål, herunder kapacitet til at producere vedvarende energi, samt netto el-import og eksport.
	• Renewable-electricity-generating-capacity-per-capita: Kapacitet til elproduktion fra vedvarende energi pr. indbygger.
	• Renewables (% equivalent primary energy): Andel af vedvarende energi i det samlede energiforbrug.
	• energy_cons_change_twh: Ændring i energiforbrug målt i terawatt-timer.
	• net_elec_imports: Netto import/eksport af elektricitet.
	• net_elec_imports_share_demand: Netto el-import som andel af el-efterspørgslen.
