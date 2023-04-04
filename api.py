from flask import Flask, request, jsonify
# from birdnet import *
import argparse
import operator
import librosa
import numpy as np
import math
import time


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import h5py


try:
    import tflite_runtime.interpreter as tflite
except:
    from tensorflow import lite as tflite

app = Flask(__name__)



# Load the trained model and label mapping
model = load_model('model/BC_1.h5')
lab = {0: 'Malacocincla abbotti_ABBOTTS BABBLER', 1: 'Papasula abbotti_ABBOTTS BOOBY', 2: 'Bucorvus abyssinicus_ABYSSINIAN GROUND HORNBILL', 3: 'Balearica regulorum_AFRICAN CROWNED CRANE', 4: 'Chrysococcyx cupreus_AFRICAN EMERALD CUCKOO', 5: 'Lagonosticta rubricata_AFRICAN FIREFINCH', 6: 'Haematopus moquini_AFRICAN OYSTER CATCHER', 7: 'Tockus fasciatus_AFRICAN PIED HORNBILL', 8: 'Diomedeidae_ALBATROSS', 9: 'Pipilo aberti_ALBERTS TOWHEE', 10: 'Psittacula eupatria_ALEXANDRINE PARAKEET', 11: 'Pyrrhocorax graculus_ALPINE CHOUGH', 12: 'Geothlypis flavovelata_ALTAMIRA YELLOWTHROAT', 13: 'Psittacula eupatria_AMERICAN AVOCET', 14: 'Botaurus lentiginosus_AMERICAN BITTERN', 15: 'Fulica americana_AMERICAN COOT', 16: 'Phoenicopterus ruber_AMERICAN FLAMINGO', 17: 'Spinus tristis_AMERICAN GOLDFINCH', 18: 'Falco sparverius_AMERICAN KESTREL', 19: 'Anthus rubescens_AMERICAN PIPIT', 20: 'Setophaga ruticilla_AMERICAN REDSTART', 21: 'Mareca americana_AMERICAN WIGEON', 22: 'Calliphlox amethystina_AMETHYST WOODSTAR', 23: 'Chloephaga melanoptera_ANDEAN GOOSE', 24: 'Vanellus resplendens_ANDEAN LAPWING', 25: 'Spinus spinescens_ANDEAN SISKIN', 26: 'Anhinga anhinga_ANHINGA', 27: 'Magumma parva_ANIANIAU', 28: 'Calypte anna_ANNAS HUMMINGBIRD', 29: 'Thamnophilidae_ANTBIRD', 30: 'Euphonia musica_ANTILLEAN EUPHONIA', 31: 'Himatione sanguinea_APAPANE', 32: 'Struthidea cinerea_APOSTLEBIRD', 33: 'Antilophia bokermanni_ARARIPE MANAKIN', 34: 'Oceanodroma homochroa_ASHY STORM PETREL', 35: 'Geokichla cinerea_ASHY THRUSHBIRD', 36: 'Nipponia nippon_ASIAN CRESTED IBIS', 37: 'Eurystomus orientalis_ASIAN DOLLARD BIRD', 38: 'Leucocarbo colensoi_AUCKLAND SHAQ', 39: 'Asthenes anthoides_AUSTRAL CANASTERO', 40: 'Sphecotheres vieilloti_AUSTRALASIAN FIGBIRD', 41: 'Amandava amandava_AVADAVAT', 42: 'Synallaxis azarae_AZARAS SPINETAIL', 43: 'Pitta steerii_AZURE BREASTED PITTA', 44: 'Cyanocorax caeruleus_AZURE JAY', 45: 'Thraupis cyanoptera_AZURE TANAGER', 46: 'Cyanistes cyanus_AZURE TIT', 47: 'Anas formosa_BAIKAL TEAL', 48: 'Haliaeetus leucocephalus_BALD EAGLE', 49: 'Geronticus eremita_BALD IBIS', 50: 'Leucopsar rothschildi_BALI STARLING', 51: 'Icterus galbula_BALTIMORE ORIOLE', 52: 'Coereba flaveola_BANANAQUIT', 53: 'Penelope argyrotis_BAND TAILED GUAN', 54: 'Eurylaimus javanicus_BANDED BROADBILL', 55: 'Hydrornis guajana_BANDED PITA', 56: 'Cladorhynchus leucocephalus_BANDED STILT', 57: 'Limosa lapponica_BAR-TAILED GODWIT', 58: 'Tyto alba_BARN OWL', 59: 'Hirundo rustica_BARN SWALLOW', 60: 'Nystalus radiatus_BARRED PUFFBIRD', 61: 'Bucephala islandica_BARROWS GOLDENEYE', 62: 'Setophaga castanea_BAY-BREASTED WARBLER', 63: 'Lybius dubius_BEARDED BARBET', 64: 'Procnias averano_BEARDED BELLBIRD', 65: 'Panurus biarmicus_BEARDED REEDLING', 66: 'Megaceryle alcyon_BELTED KINGFISHER', 67: 'Strelitzia_BIRD OF PARADISE', 68: 'Eurylaimus ochromalus_BLACK & YELLOW BROADBILL', 69: 'Aviceda leuphotes_BLACK BAZA', 70: 'Calyptorhynchus latirostris_BLACK COCKATO', 71: 'Francolinus francolinus_BLACK FRANCOLIN', 72: 'Rynchops niger_BLACK SKIMMER', 73: 'Cygnus atratus_BLACK SWAN', 74: 'Amaurornis bicolor_BLACK TAIL CRAKE', 75: 'Aegithalos concinnus_BLACK THROATED BUSHTIT', 76: 'Setophaga caerulescens_BLACK THROATED WARBLER', 77: 'Puffinus opisthomela_BLACK VENTED SHEARWATER', 78: 'Coragyps atratus_BLACK VULTURE', 79: 'Poecile atricapillus_BLACK-CAPPED CHICKADEE', 80: 'Podiceps nigricollis_BLACK-NECKED GREBE', 81: 'Amphispiza bilineata_BLACK-THROATED SPARROW', 82: 'Setophaga fusca_BLACKBURNIAM WARBLER', 83: 'Celeus flavescens_BLONDE CRESTED WOODPECKER', 84: 'Ithaginis cruentus_BLOOD PHEASANT', 85: 'Coua caerulea_BLUE COAU', 86: 'Dacnis cayana_BLUE DACNIS', 87: 'Dendragapus obscurus_BLUE GROUSE', 88: 'Ardea herodias_BLUE HERON', 89: 'Ceuthmochares aereus_BLUE MALKOHA', 90: 'Aulacorhynchus caeruleogularis_BLUE THROATED TOUCANET', 91: 'Dolichonyx oryzivorus_BOBOLINK', 92: 'Pityriasis gymnocephala_BORNEAN BRISTLEHEAD', 93: 'Chloropsis kinabaluensis_BORNEAN LEAFBIRD', 94: 'Polyplectron schleiermacheri_BORNEAN PHEASANT', 95: 'Phalacrocorax penicillatus_BRANDT CORMARANT', 96: 'Euphagus cyanocephalus_BREWERS BLACKBIRD', 97: 'Certhia americana_BROWN CREPPER', 98: 'Anous stolidus_BROWN NOODY', 99: 'Toxostoma rufum_BROWN THRASHER', 100: 'Bucephala albeola_BUFFLEHEAD', 101: 'Lophura bulweri_BULWERS PHEASANT', 102: 'Cursorius rufus_BURCHELLS COURSER', 103: 'Alectura lathami_BUSH TURKEY', 104: 'Pseudoseisura cristat_CAATINGA CACHOLOTE', 105: 'Campylorhynchus brunneicapillus_CACTUS WREN', 106: 'Gymnogyps californianus_CALIFORNIA CONDOR', 107: 'Larus californicus_CALIFORNIA GULL', 108: 'Callipepla californica_CALIFORNIA QUAIL', 109: 'Colaptes campestris_CAMPO FLICKER', 110: 'Serinus canaria domestica_CANARY', 111: 'Lamprotornis nitens_CAPE GLOSSY STARLING', 112: 'Macronyx capensis_CAPE LONGCLAW', 113: 'Setophaga tigrina_CAPE MAY WARBLER', 114: 'Monticola rupestris_CAPE ROCK THRUSH', 115: 'Pilherodius pileatus_CAPPED HERON', 116: 'Perissocephalus tricolor_CAPUCHINBIRD', 117: 'Merops nubicoides_CARMINE BEE-EATER', 118: 'Hydroprogne caspia_CASPIAN TERN', 119: 'Casuarius_CASSOWARY', 120: 'Bombycilla cedrorum_CEDAR WAXWING', 121: 'Setophaga cerulea_CERULEAN WARBLER', 122: 'Cyanolyca armillata_CHARA DE COLLAR', 123: 'Lorius garrulus_CHATTERING LORY', 124: 'Euphonia pectoralis_CHESTNET BELLIED EUPHONIA', 125: 'Bambusicola thoracicus_CHINESE BAMBOO PARTRIDGE', 126: 'Ardeola bacchus_CHINESE POND HERON', 127: 'Spizella passerina_CHIPPING SPARROW', 128: 'Scelorchilus rubecula_CHUCAO TAPACULO', 129: 'Alectoris chukar_CHUKAR PARTRIDGE', 130: 'Attila cinnamomeus_CINNAMON ATTILA', 131: 'Pyrrhomyias cinnamomeus_CINNAMON FLYCATCHER', 132: 'Anas cyanoptera_CINNAMON TEAL', 133: 'Nucifraga columbiana_CLARKS NUTCRACKER', 134: 'Rupicola_COCK OF THE  ROCK', 135: 'Cacatuidae_COCKATOO', 136: 'Pteroglossus torquatus_COLLARED ARACARI', 137: 'Regulus ignicapilla_COMMON FIRECREST', 138: 'Quiscalus quiscula_COMMON GRACKLE', 139: 'Delichon urbicum_COMMON HOUSE MARTIN', 140: 'Aegithina tiphia_COMMON IORA', 141: 'Gavia immer_COMMON LOON', 142: 'Phalaenoptilus nuttallii_COMMON POORWILL', 143: 'Sturnus vulgaris_COMMON STARLING', 144: 'Centropus cupreicaudus_COPPERY TAILED COUCAL', 145: 'Dromas ardeola_CRAB PLOVER', 146: 'Geranospiza caerulescens_CRANE HAWK', 147: 'Celeus flavus_CREAM COLORED WOODPECKER', 148: 'Aethia cristatella_CRESTED AUKLET', 149: 'Caracara cheriway_CRESTED CARACARA', 150: 'Coua cristata_CRESTED COUA', 151: 'Lophura ignita_CRESTED FIREBACK', 152: 'Megaceryle lugubris_CRESTED KINGFISHER', 153: 'Sitta carolinensis_CRESTED NUTHATCH', 154: 'Psarocolius decumanus_CRESTED OROPENDOLA', 155: 'Falcunculus frontatus_CRESTED SHRIKETIT', 156: 'Epthianura tricolor_CRIMSON CHAT', 157: 'Aethopyga siparaja_CRIMSON SUNBIRD', 158: 'Corvus_CROW', 159: 'Goura_CROWNED PIGEON', 160: 'Todus multicolor_CUBAN TODY', 161: 'Priotelus temnurus_CUBAN TROGON', 162: 'Pteroglossus beauharnaesii_CURL CRESTED ARACURI', 163: 'Trachyphonus darnaudii_D-ARNAUDS BARBET', 164: 'Pelecanus crispus_DALMATIAN PELICAN', 165: 'Dendrocopos darjellensis_DARJEELING WOODPECKER', 166: 'Junco hyemalis_DARK EYED JUNCO', 167: 'Pyrocephalus nanus_DARWINS FLYCATCHER', 168: 'Phoenicurus auroreus_DAURIAN REDSTART', 169: 'Grus virgo_DEMOISELLE CRANE', 170: 'Taeniopygia bichenovii_DOUBLE BARRED FINCH', 171: 'Phalacrocorax auritus_DOUBLE BRESTED CORMARANT', 172: 'Cyclopsitta diophthalma_DOUBLE EYED FIG PARROT', 173: 'Picoides pubescens_DOWNY WOODPECKER', 174: 'Pseudeos fuscata_DUSKY LORY', 175: 'Melanodryas vittata_DUSKY ROBIN', 176: 'Hydrornis phayrei_EARED PITA', 177: 'Sialia sialis_EASTERN BLUEBIRD', 178: 'Northiella haematogaster_EASTERN BLUEBONNET', 179: 'Ploceus subaureus_EASTERN GOLDEN WEAVER', 180: 'Sturnella magna_EASTERN MEADOWLARK', 181: 'Platycercus eximius_EASTERN ROSELLA', 182: 'Pipilo erythrophthalmus_EASTERN TOWEE', 183: 'Antrostomus vociferus_EASTERN WIP POOR WILL', 184: 'Oreotrochilus chimborazo_ECUADORIAN HILLSTAR', 185: 'Alopochen aegyptiaca_EGYPTIAN GOOSE', 186: 'Trogon elegans_ELEGANT TROGON', 187: 'Syrmaticus ellioti_ELLIOTS  PHEASANT', 188: 'Tangara florida_EMERALD TANAGER', 189: 'Aptenodytes forsteri_EMPEROR PENGUIN', 190: 'Dromaius novaehollandiae_EMU', 191: 'Gracula enganensis_ENGGANO MYNA', 192: 'Pyrrhula pyrrhula_EURASIAN BULLFINCH', 193: 'Oriolus oriolus_EURASIAN GOLDEN ORIOLE', 194: 'Pica pica_EURASIAN MAGPIE', 195: 'Carduelis carduelis_EUROPEAN GOLDFINCH', 196: 'Streptopelia turtur_EUROPEAN TURTLE DOVE', 197: 'Coccothraustes vespertinus_EVENING GROSBEAK', 198: 'Irena_FAIRY BLUEBIRD', 199: 'Eudyptula minor_FAIRY PENGUIN', 200: 'Sternula nereis_FAIRY TERN', 201: 'Euplectes axillaris_FAN TAILED WIDOW', 202: 'Campylorhynchus fasciatus_FASCIATED WREN', 203: 'Pericrocotus igneus_FIERY MINIVET', 204: 'Eudyptes pachyrhynchus_FIORDLAND PENGUIN', 205: 'Myzornis pyrrhoura_FIRE TAILLED MYZORNIS', 206: 'Sericulus ardens_FLAME BOWERBIRD', 207: 'Piranga bidentata_FLAME TANAGER', 208: 'Fregatidae_FRIGATE', 209: 'Callipepla gambelii_GAMBELS QUAIL', 210: 'Callocephalon fimbriatum_GANG GANG COCKATOO', 211: 'Melanerpes uropygialis_GILA WOODPECKER', 212: 'Colaptes auratus_GILDED FLICKER', 213: 'Plegadis falcinellus_GLOSSY IBIS', 214: 'Corythaixoides concolor_GO AWAY BIRD', 215: 'Vermivora chrysoptera_GOLD WING WARBLER', 216: 'Prionodura newtoniana_GOLDEN BOWER BIRD', 217: 'Setophaga chrysoparia_GOLDEN CHEEKED WARBLER', 218: 'Chlorophonia callophrys_GOLDEN CHLOROPHONIA', 219: 'Aquila chrysaetos_GOLDEN EAGLE', 220: 'Guaruba guarouba_GOLDEN PARAKEET', 221: 'Chrysolophus pictus_GOLDEN PHEASANT', 222: 'Tmetothylacus tenellus_GOLDEN PIPIT', 223: 'Erythrura gouldiae_GOULDIAN FINCH', 224: 'Grandala coelicolor_GRANDALA', 225: 'Dumetella carolinensis_GRAY CATBIRD', 226: 'Tyrannus dominicensis_GRAY KINGBIRD', 227: 'Perdix perdix_GRAY PARTRIDGE', 228: 'Strix nebulosa_GREAT GRAY OWL', 229: 'Jacamerops aureus_GREAT JACAMAR', 230: 'Pitangus sulphuratus_GREAT KISKADEE', 231: 'Nyctibius grandis_GREAT POTOO', 232: 'Tinamus major_GREAT TINAMOU', 233: 'Megaxenops parnaguae_GREAT XENOPS', 234: 'Contopus pertinax_GREATER PEWEE', 235: 'Centrocercus urophasianus_GREATOR SAGE GROUSE', 236: 'Calyptomena viridis_GREEN BROADBILL', 237: 'Cyanocorax yncas_GREEN JAY', 238: 'Cissa chinensis_GREEN MAGPIE', 239: 'Coracina caesia_GREY CUCKOOSHRIKE', 240: 'Pluvialis squatarola_GREY PLOVER', 241: 'Crotophaga sulcirostris_GROVED BILLED ANI', 242: 'Tauraco persa_GUINEA TURACO', 243: 'Numididae_GUINEAFOWL', 244: 'Hydrornis gurneyi_GURNEYS PITTA', 245: 'Falco rusticolus_GYRFALCON', 246: 'Scopus umbretta_HAMERKOP', 247: 'Histrionicus histrionicus_HARLEQUIN DUCK', 248: 'Coturnix delegorguei_HARLEQUIN QUAIL', 249: 'Harpia harpyja_HARPY EAGLE', 250: 'Branta sandvicensis_HAWAIIAN GOOSE', 251: 'Coccothraustes coccothraustes_HAWFINCH', 252: 'Euryceros prevostii_HELMET VANGA', 253: 'Piranga flava_HEPATIC TANAGER', 254: 'Tarsiger rufilatus_HIMALAYAN BLUETAIL', 255: 'Lophophorus impejanus_HIMALAYAN MONAL', 256: 'Ophisthocomus hoazin_HOATZIN', 257: 'Lophodytes cucullatus_HOODED MERGANSER', 258: 'Upupidae_HOOPOES', 259: 'Oreophasis derbianus_HORNED GUAN', 260: 'Eremophila alpestris_HORNED LARK', 261: 'Heliactin bilophus_HORNED SUNGEM', 262: 'Haemorhous mexicanus_HOUSE FINCH', 263: 'Passer domesticus_HOUSE SPARROW', 264: 'Anodorhynchus hyacinthinus_HYACINTH MACAW', 265: 'Cyanopica cooki_IBERIAN MAGPIE', 266: 'Ibidorhyncha struthersii_IBISBILL', 267: 'Leucocarbo atriceps_IMPERIAL SHAQ', 268: 'Larosterna inca_INCA TERN', 269: 'Ardeotis nigriceps_INDIAN BUSTARD', 270: 'Pitta brachyura_INDIAN PITTA', 271: 'Coracias benghalensis_INDIAN ROLLER', 272: 'Gyps indicus_INDIAN VULTURE', 273: 'Passerina cyanea_INDIGO BUNTING', 274: 'Eumyias indigo_INDIGO FLYCATCHER', 275: 'Charadrius australis_INLAND DOTTEREL', 276: 'Pteroglossus azara_IVORY BILLED ARACARI', 277: 'Pagophila eburnea_IVORY GULL', 278: ' Vestiaria coccinea._IWI', 279: 'Jabiru mycteria_JABIRU', 280: 'Lymnocryptes minimus_JACK SNIPE', 281: 'Aratinga jandaya_JANDAYA PARAKEET', 282: 'Erithacus akahige_JAPANESE ROBIN', 283: 'Lonchura oryzivora_JAVA SPARROW', 284: 'Grallaria ridgelyi_JOCOTOCO ANTPITTA', 285: 'Rhynochetos jubatus_KAGU', 286: 'Strigops habroptilus_KAKAPO', 287: 'Charadrius vociferus_KILLDEAR', 288: 'Somateria spectabilis_KING EIDER', 289: 'Sarcoramphus papa_KING VULTURE', 290: 'Apteryx_KIWI', 291: 'Dacelo_KOOKABURRA', 292: 'Calamospiza melanocorys_LARK BUNTING', 293: 'Passerina amoena_LAZULI BUNTING', 294: 'Leptoptilos javanicus_LESSER ADJUTANT', 295: 'Coracias caudatus_LILAC ROLLER', 296: 'Alle alle_LITTLE AUK', 297: 'Lanius ludovicianus_LOGGERHEAD SHRIKE', 298: 'Asio otus_LONG-EARED OWL', 299: 'Anseranas semipalmata_MAGPIE GOOSE', 300: 'Ocyceros griseus_MALABAR HORNBILL', 301: 'Corythornis cristatus_MALACHITE KINGFISHER', 302: 'Zosterops maderaspatanus_MALAGASY WHITE EYE', 303: 'Macrocephalon maleo_MALEO', 304: 'Anas platyrhynchos_MALLARD DUCK', 305: 'Aix galericulata_MANDRIN DUCK', 306: 'Coccyzus minor_MANGROVE CUCKOO', 307: 'Leptoptilos crumenifer_MARABOU STORK', 308: 'Sula dactylatra_MASKED BOOBY', 309: 'Vanellus miles_MASKED LAPWING', 310: 'Plectrophenax hyperboreus_MCKAYS BUNTING', 311: 'Syrmaticus mikado_MIKADO  PHEASANT', 312: 'Zenaida macroura_MOURNING DOVE', 313: 'Acridotheres tristis_MYNA', 314: 'Caloenas nicobarica_NICOBAR PIGEON', 315: 'Philemon corniculatus_NOISY FRIARBIRD', 316: 'Camptostoma imberbe_NORTHERN BEARDLESS TYRANNULET', 317: 'Cardinalis cardinalis_NORTHERN CARDINAL', 318: 'Colaptes auratus_NORTHERN FLICKER', 319: 'Fulmarus glacialis_NORTHERN FULMAR', 320: 'Morus bassanus_NORTHERN GANNET', 321: 'Accipiter gentilis_NORTHERN GOSHAWK', 322: 'Jacana spinosa_NORTHERN JACANA', 323: 'Mimus polyglottos_NORTHERN MOCKINGBIRD', 324: 'Setophaga americana_NORTHERN PARULA', 325: 'Euplectes franciscanus_NORTHERN RED BISHOP', 326: 'Spatula clypeata_NORTHERN SHOVELER', 327: 'Meleagris ocellata_OCELLATED TURKEY', 328: 'Gallirallus okinawae_OKINAWA RAIL', 329: 'Passerina leclancherii_ORANGE BRESTED BUNTING', 330: 'Phodilus badius_ORIENTAL BAY OWL', 331: 'Pandion haliaetus_OSPREY', 332: 'Struthio camelus_OSTRICH', 333: 'Seiurus aurocapilla_OVENBIRD', 334: 'Haematopus_OYSTER CATCHER', 335: 'Passerina ciris_PAINTED BUNTING', 336: 'Loxioides bailleui_PALILA', 337: 'Tangara chilensis_PARADISE TANAGER', 338: 'Aethia psittacula_PARAKETT  AKULET', 339: 'Parus major_PARUS MAJOR', 340: 'Phrygilus patagonicus_PATAGONIAN SIERRA FINCH', 341: 'Pavo cristatus_PEACOCK', 342: 'Falco peregrinus_PEREGRINE FALCON', 343: 'Pithecophaga jefferyi_PHILIPPINE EAGLE', 344: 'Petroica rodinogaster_PINK ROBIN', 345: 'Stercorarius pomarinus_POMARINE JAEGER', 346: 'Fratercula_PUFFIN', 347: 'Haemorhous purpureus_PURPLE FINCH', 348: 'Porphyrio martinicus_PURPLE GALLINULE', 349: 'Progne subis_PURPLE MARTIN', 350: 'Porphyrio porphyrio_PURPLE SWAMPHEN', 351: 'Ispidina picta_PYGMY KINGFISHER', 352: 'Pharomachrus mocinno_QUETZAL', 353: 'Trichoglossus moluccanus_RAINBOW LORIKEET', 354: 'Alca torda_RAZORBILL', 355: 'a_STRAWBERRY FINCH', 406: 'Asio clamator_STRIPED OWL', 407: 'Machaeropterus regulus_STRIPPED MANAKIN', 408: 'Cecropis abyssinica_STRIPPED SWALLOW', 409: 'Lamprotornis superbus_SUPERB STARLING', 410: 'Lophura swinhoii_SWINHOES PHEASANT', 411: 'Orthotomus atrogularis_TAILORBIRD', 412: 'Urocissa caerulea_TAIWAN MAGPIE', 413: 'Porphyrio hochstetteri_TAKAHE', 414: 'Tribonyx mortierii_TASMANIAN HEN', 415: 'Anas crecca_TEAL DUCK', 416: 'Paridae_TIT MOUSE', 417: 'Ramphastidae_TOUCHAN', 418: 'Setophaga townsendi_TOWNSENDS WARBLER', 419: 'Tachycineta bicolor_TREE SWALLOW', 420: 'Agelaius tricolor_TRICOLORED BLACKBIRD', 421: 'Tyrannus melancholicus_TROPICAL KINGBIRD', 422: 'Cygnus buccinator_TRUMPTER SWAN', 423: 'Cathartes aura_TURKEY VULTURE', 424: 'Eumomota superciliosa_TURQUOISE MOTMOT', 425: 'Cephalopterus_UMBRELLA BIRD', 426: 'Ixoreus naevius_VARIED THRUSH', 427: 'Catharus fuscescens_VEERY', 428: 'Icterus icterus_VENEZUELIAN TROUPIAL', 429: 'Pyrocephalus obscurus_VERMILION FLYCATHER', 430: 'Goura victoria_VICTORIA CROWNED PIGEON', 431: 'Tachycineta thalassina_VIOLET GREEN SWALLOW', 432: 'Musophaga violacea_VIOLET TURACO', 433: 'Acryllium vulturinum_VULTURINE GUINEAFOWL', 434: 'Tichodroma muraria_WALL CREAPER', 435: 'Crax globulosa_WATTLED CURASSOW', 436: 'Vanellus indicus_WATTLED LAPWING', 437: 'Numenius phaeopus_WHIMBREL', 438: 'Amaurornis cinerea_WHITE BROWED CRAKE', 439: 'Tauraco leucotis_WHITE CHEEKED TURACO', 440: 'Tropicranus albocristatus_WHITE CRESTED HORNBILL', 441: 'Corvus albicollis_WHITE NECKED RAVEN', 442: 'Phaethon lepturus_WHITE TAILED TROPIC', 443: 'Merops albicollis_WHITE THROATED BEE EATER', 444: 'Meleagris gallopavo_WILD TURKEY', 445: 'Cicinnurus respublica_WILSONS BIRD OF PARADISE', 446: 'Aix sponsa_WOOD DUCK', 447: 'Dicaeum melanoxanthum_YELLOW BELLIED FLOWERPECKER', 448: 'Cacicus cela_YELLOW CACIQUE', 449: 'Xanthocephalus_YELLOW HEADED BLACKBIRD'}

# Initialize the Flask app
app = Flask(__name__)

# Define a route for the image recognition API
@app.route('/recognize', methods=['POST'])
def recognize_image():
    # Load and preprocess the image from the request
    file = request.files['image']
    file_path = 'temp.jpg'
    file.save(file_path)
    
    img = load_img(file_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    
    # Use the model to predict the class and probability of the image
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)[0]
    y_prob = answer[0][y_class]
    res = lab[y_class]
    
    # Return the result as a JSON object
    result = [{
        'class': res.split("_")[1],
        'scientific_class': res.split("_")[0],
        'probability': float(y_prob)
    }]
    return jsonify({'images':result})





def readAudioData(path, overlap, sample_rate=48000):

    print('READING AUDIO DATA...', end=' ', flush=True)

    # Open file with librosa (uses ffmpeg or libav)
    sig, rate = librosa.load(path, sr=sample_rate, mono=True, res_type='kaiser_fast')

    # Split audio into 3-second chunks
    chunks = splitSignal(sig, rate, overlap)

    print('DONE! READ', str(len(chunks)), 'CHUNKS.')

    return chunks


def loadModel():

    global INPUT_LAYER_INDEX
    global OUTPUT_LAYER_INDEX
    global MDATA_INPUT_INDEX
    global CLASSES

    print('LOADING TF LITE MODEL...', end=' ')

    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path='model/BirdNET_6K_GLOBAL_MODEL.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get input tensor index
    INPUT_LAYER_INDEX = input_details[0]['index']
    MDATA_INPUT_INDEX = input_details[1]['index']
    OUTPUT_LAYER_INDEX = output_details[0]['index']

    # Load labels
    CLASSES = []
    with open('model/labels.txt', 'r') as lfile:
        for line in lfile.readlines():
            CLASSES.append(line.replace('\n', ''))

    print('DONE!')

    return interpreter

def splitSignal(sig, rate, overlap, seconds=3.0, minlen=1.5):

    # Split signal with overlap
    sig_splits = []
    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)]

        # End of signal?
        if len(split) < int(minlen * rate):
            break
        
        # Signal chunk too short? Fill with zeros.
        if len(split) < int(rate * seconds):
            temp = np.zeros((int(rate * seconds)))
            temp[:len(split)] = split
            split = temp
        
        sig_splits.append(split)

    return sig_splits

def analyzeAudioData(chunks, lat, lon, week, sensitivity, overlap, interpreter):

    detections = {}
    start = time.time()
    print('ANALYZING AUDIO...', end=' ', flush=True)

    # Convert and prepare metadata
    mdata = convertMetadata(np.array([lat, lon, week]))
    mdata = np.expand_dims(mdata, 0)

    # Parse every chunk
    pred_start = 0.0
    for c in chunks:

        # Prepare as input signal
        sig = np.expand_dims(c, 0)

        # Make prediction
        p = predict([sig, mdata], interpreter, sensitivity)

        # Save result and timestamp
        pred_end = pred_start + 3.0
        detections[str(pred_start) + ';' + str(pred_end)] = p
        pred_start = pred_end - overlap

    print('DONE! Time', int((time.time() - start) * 10) / 10.0, 'SECONDS')

    return detections


def custom_sigmoid(x, sensitivity=1.0):
    return 1 / (1.0 + np.exp(-sensitivity * x))

def predict(sample, interpreter, sensitivity):

    # Make a prediction
    interpreter.set_tensor(INPUT_LAYER_INDEX, np.array(sample[0], dtype='float32'))
    interpreter.set_tensor(MDATA_INPUT_INDEX, np.array(sample[1], dtype='float32'))
    interpreter.invoke()
    prediction = interpreter.get_tensor(OUTPUT_LAYER_INDEX)[0]

    # Apply custom sigmoid
    p_sigmoid = custom_sigmoid(prediction, sensitivity)

    # Get label and scores for pooled predictions
    p_labels = dict(zip(CLASSES, p_sigmoid))

    # Sort by score
    p_sorted = sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)

    # Remove species that are on blacklist
    for i in range(min(10, len(p_sorted))):
        if p_sorted[i][0] in ['Human_Human', 'Non-bird_Non-bird', 'Noise_Noise']:
            p_sorted[i] = (p_sorted[i][0], 0.0)

    # Only return first the top ten results
    return p_sorted[:10]


def convertMetadata(m):

    # Convert week to cosine
    if m[2] >= 1 and m[2] <= 48:
        m[2] = math.cos(math.radians(m[2] * 7.5)) + 1 
    else:
        m[2] = -1

    # Add binary mask
    mask = np.ones((3,))
    if m[0] == -1 or m[1] == -1:
        mask = np.zeros((3,))
    if m[2] == -1:
        mask[2] = 0.0

    return np.concatenate([m, mask])

def writeResultsToFile(detections, min_conf):
    # white_list_dict = {k: True for k in WHITE_LIST}
    print('WRITING RESULTS...', end=' ')
    rcnt = 0
    results_list = []
    with open('results.csv', 'w') as rfile:
        head = 'Start (s);End (s);Scientific name;Common name;Confidence'
        rfile.write(head + '\n')
        headers = head.strip().split(';')
        for d in detections:
            for entry in detections[d]:
                if entry[1] >= min_conf:
                    result_str =d + ';' + entry[0].replace('_', ';') + ';' + str(entry[1])
                    rfile.write(result_str + '\n')
                    print("\n"+result_str)
                    fields = result_str.split(';')
                    result_dict = {
                        headers[0]: float(fields[0]),
                        headers[1]: float(fields[1]),
                        headers[2]: fields[2],
                        headers[3]: fields[3],
                        headers[4]: float(fields[4])
                    }
                    results_list.append(result_dict)
                    rcnt += 1
    print('DONE! WROTE', rcnt, 'RESULTS.')
    return jsonify({'audio': results_list})


interpreter = loadModel()


@app.route('/', methods=['GET'])
def hello():
    return 'Hello, World!'

@app.route('/api/predicts', methods=['POST'])
def predict_species():
    # Load audio data from request
    audio_file = request.files['audio']
    audio_file_path = 'temp.mp3'
    audio_file.save(audio_file_path)

    lat = request.form.get('lat', -1)
    lon = request.form.get('lon', -1)
    week = request.form.get('week', -1)
    overlap = float(request.form.get('overlap', '0.5'))
    sensitivity = float(request.form.get('sensitivity', '1.0'))
    min_conf = float(request.form.get('min_conf','0.1'))

    # Read audio data and split into chunks
    audio_chunks = readAudioData(audio_file_path, overlap)

    # Analyze audio data
    detections = analyzeAudioData(audio_chunks, lat, lon, week, sensitivity, overlap, interpreter)

    # Return JSON response with detections

     # Write detections to output file
    min_conf = max(0.01, min(min_conf, 0.99))
    output = writeResultsToFile(detections, min_conf)
    return output

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
